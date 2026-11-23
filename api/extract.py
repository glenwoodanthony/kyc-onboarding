"""Structure OCR text into a validated KYC document with a local model.

`extract(ocr, declared_type, country)` sends the OCR text to the structuring model
(Ollama's OpenAI-compatible endpoint, through Instructor) with the system prompt in
prompts/system_prompt.md, and validates the answer against the declared type's model
in api/schemas.py.

- The answer is validated against the declared type only, which keeps the schema
  small for a small model. Validation errors go back to the model for another try,
  up to MAX_RETRIES times.
- Model failures never raise. If no valid document comes back (bad output on every
  attempt, Ollama down, timeout), the result is the declared type with every field
  at confidence "none" and an EXTRACTION_FAILED flag.
- A missing setting (OLLAMA_BASE_URL, STRUCTURING_MODEL in .env or the environment)
  does raise: that's a setup error, not a model failure.
"""

import json
import logging
from functools import cache
from pathlib import Path
from typing import Any

import instructor
import openai
from instructor.core.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
    ResponseParsingError,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from api.ocr import OcrResult
from api.schemas import (
    DOCUMENT_MODELS,
    Confidence,
    DocumentType,
    Flag,
    FlagCode,
    KYCExtraction,
    field_names,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")

# Retries after the first attempt. Instructor's max_retries counts retries, so the
# model is called at most MAX_RETRIES + 1 times.
MAX_RETRIES = 2

# Turn the model's thinking off. On qwen3:4b, thinking made one attempt take many
# minutes and eats into Ollama's default 4096-token context, which the prompt and
# schema already fill to about 3300 tokens. Without it a passport takes about 30 s.
REASONING_EFFORT = "none"

# Failures that mean "the model gave no usable document", as opposed to bugs.
MODEL_FAILURES = (
    InstructorRetryException,
    IncompleteOutputException,
    ResponseParsingError,
    openai.OpenAIError,
)


class Settings(BaseSettings):
    """Read from the environment, then from <repo>/.env. No defaults: set both."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    ollama_base_url: str
    structuring_model: str


@cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue] - fields come from env


def _openai_client(settings: Settings) -> openai.OpenAI:
    # Ollama ignores the key, but the SDK requires one.
    return openai.OpenAI(base_url=settings.ollama_base_url, api_key="ollama")


def format_user_message(
    ocr: OcrResult, declared_type: DocumentType, country: str
) -> str:
    """The request in the format prompts/system_prompt.md describes."""
    parts = [
        f"<declared_type>{declared_type}</declared_type>",
        f"<country>{country}</country>",
    ]
    if ocr.engine == "pp-ocr":
        # PP-OCR markdown is exactly its lines joined, so line n is ocr.lines[n-1].
        texts = [line.text for line in ocr.lines]
        scores = [
            {"line": n, "confidence": round(line.confidence, 2)}
            for n, line in enumerate(ocr.lines, start=1)
            if line.confidence is not None
        ]
    else:
        # VL markdown has no per-line scores and doesn't map 1:1 onto ocr.lines.
        texts = [text for text in ocr.markdown.splitlines() if text.strip()]
        scores = []
    numbered = "\n".join(f"L{n}: {text}" for n, text in enumerate(texts, start=1))
    parts.append(f"<ocr_markdown>\n{numbered}\n</ocr_markdown>")
    if scores:
        parts.append(f"<line_confidence>\n{json.dumps(scores)}\n</line_confidence>")
    return "\n".join(parts)


def extract(
    ocr: OcrResult,
    declared_type: DocumentType,
    country: str,
    *,
    settings: Settings | None = None,
) -> KYCExtraction:
    """Extract one document of `declared_type` from OCR text. Doesn't raise on model failure."""
    settings = settings or get_settings()
    model_cls = DOCUMENT_MODELS[declared_type]
    client = instructor.from_openai(_openai_client(settings), mode=instructor.Mode.JSON)

    try:
        doc = client.chat.completions.create(
            model=settings.structuring_model,
            response_model=model_cls,
            max_retries=MAX_RETRIES,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_user_message(ocr, declared_type, country),
                },
            ],
        )
    except MODEL_FAILURES as exc:
        reason = f"extraction failed: {type(exc).__name__}: {exc}"
        logger.warning("%s extraction failed: %s", declared_type, reason)
        return _failed(declared_type, country, reason)

    # Rebuild from the fields alone, so the only flags are the ones code sets.
    data: dict[str, Any] = doc.model_dump(exclude={"flags"})
    flags: list[Flag] = []
    if all(
        getattr(doc, name).confidence is Confidence.NONE
        for name in field_names(model_cls)
    ):
        flags.append(
            Flag(
                code=FlagCode.DOCUMENT_TYPE_MISMATCH,
                message=f"no fields found; the document may not be a {declared_type}",
            )
        )
    return _build(declared_type, country, data, flags)


def _failed(declared_type: DocumentType, country: str, reason: str) -> KYCExtraction:
    """Every field of the declared type at confidence "none", flagged EXTRACTION_FAILED."""
    data = {
        name: {"value": None, "raw_text": None, "confidence": "none", "note": reason}
        for name in field_names(DOCUMENT_MODELS[declared_type])
    }
    flag = Flag(code=FlagCode.EXTRACTION_FAILED, message=reason)
    return _build(declared_type, country, data, [flag])


def _build(
    declared_type: DocumentType,
    country: str,
    data: dict[str, Any],
    flags: list[Flag],
) -> KYCExtraction:
    document = {
        **data,
        "document_type": declared_type,
        "flags": [f.model_dump() for f in flags],
    }
    return KYCExtraction.model_validate(
        {"declared_type": declared_type, "country": country, "document": document}
    )
