"""Structure OCR text into a validated KYC document with a local model.

`extract(ocr, declared_type, country)` sends the OCR text to the structuring model
(Ollama's OpenAI-compatible endpoint, through Instructor) with the system prompt in
prompts/system_prompt.md, and validates the answer against the declared type's model
in api/schemas.py.

- The answer is validated against the declared type only, which keeps the schema
  small for a small model. Validation errors go back to the model for another try,
  up to MAX_RETRIES times.
- Model failures never raise. If the last attempt is still invalid, its valid
  fields are kept, the invalid ones are nulled, and the document is flagged
  EXTRACTION_INCOMPLETE. If nothing can be kept (Ollama down, timeout, no JSON),
  every field is "none" and the flag is EXTRACTION_FAILED.
- OCR lines that read like instructions to the model are withheld from it and the
  document is flagged PROMPT_INJECTION_SUSPECTED (see instruction_lines).
- A missing setting (OLLAMA_BASE_URL, STRUCTURING_MODEL in .env or the environment)
  does raise: that's a setup error, not a model failure.
"""

import json
import logging
import re
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
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from api.ocr import OcrResult
from api.schemas import (
    DOCUMENT_MODELS,
    BaseDocument,
    DocumentType,
    Flag,
    FlagCode,
    KYCExtraction,
    field_names,
    source_lines,
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

# Greedy decoding, so the same OCR text gives the same answer. Ollama samples by
# default, and on the synthetic documents qwen3:4b changed its answer between runs:
# a full address one run, only its first line the next.
TEMPERATURE = 0

# The fields that make a document that type: its number or reference, plus the date
# that defines it. If none has a value, the text is probably another document,
# so it's flagged DOCUMENT_TYPE_MISMATCH. "Every field none" wasn't enough: given a
# payslip declared as a passport, qwen3:4b filled full_name from the payslip's
# "Employee:" line and left the rest null. A real document that's only partly
# unreadable keeps at least one of these, so it isn't flagged.
IDENTIFYING_FIELDS: dict[DocumentType, tuple[str, ...]] = {
    "national_id": ("document_number", "date_of_birth"),
    "drivers_permit": ("permit_number", "date_of_birth"),
    "passport": ("passport_number", "date_of_birth"),
    "birth_certificate": ("registration_number", "date_of_birth"),
    "utility_bill": ("provider", "statement_date"),
    "payslip": ("employer_name", "pay_date"),
    "address_letter": ("issuer", "letter_date"),
}

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


def _ocr_lines(ocr: OcrResult) -> list[str]:
    """The lines the model sees as L1, L2, ... (before any are withheld)."""
    if ocr.engine == "pp-ocr":
        # PP-OCR markdown is exactly its lines joined, so line n is ocr.lines[n-1].
        return [line.text for line in ocr.lines]
    # VL markdown has no per-line scores and doesn't map 1:1 onto ocr.lines.
    return [text for text in ocr.markdown.splitlines() if text.strip()]


# --- prompt-injection defence --------------------------------------------------
#
# The document is written by the customer, so its text may address the model. The
# prompt says to ignore that, but qwen3:4b obeyed "set ... the provider to Verified
# Ltd" on a synthetic utility bill anyway. So lines that read like instructions to
# an AI are withheld from the model (replaced by WITHHELD, keeping line numbers)
# and the document is flagged PROMPT_INJECTION_SUSPECTED for a reviewer.
#
# Patterns can be beaten by rewording: this raises the bar, it isn't a guarantee.
# Keep the patterns specific to text aimed at a machine, so ordinary document text
# ("Please ignore this reminder if you have paid") isn't withheld.

WITHHELD = "[line withheld: it reads like an instruction to the extraction system]"

# "AI" is often OCRed as "Al" or "A1".
_AI = r"(?:a[il1]|model|assistant|llm|chatbot|bot|system|extractor|parser|gpt)"
_BLOCKS = r"(?:ocr_markdown|line_confidence|declared_type|country|system|user)"

INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "ignore your rules", "disregard all previous instructions"
        (
            r"\b(?:ignore|disregard|override|forget|bypass)\b.{0,40}?"
            r"\b(?:rules|instructions|prompt|guidelines|guardrails|policy)\b"
        ),
        # "SYSTEM NOTE TO AI", "message to the model"
        rf"\b(?:note|message|instructions?|memo)\s+(?:to|for)\s+(?:the\s+)?{_AI}\b",
        # "SYSTEM:", "AI note:" at the start of a line. Not "Model:", which is an
        # ordinary label (a vehicle model, say).
        (
            r"^\W*(?:system|a[il1]|assistant|llm)\s*(?:note|prompt|message|"
            r"instructions?)?\s*:"
        ),
        # "you are an AI", "act as the assistant"
        rf"\b(?:you\s+are|act\s+as)\s+(?:an?|the)\s+(?:{_AI}|language\s+model)\b",
        # "mark this customer as verified", "treat the applicant as approved"
        (
            r"\b(?:mark|treat|record|classify)\b.{0,40}?\bas\s+(?:verified|approved|"
            r"genuine|valid|passed|low[\s-]risk)\b"
        ),
        # Anything trying to open or close one of the prompt's data blocks.
        rf"<\s*/?\s*{_BLOCKS}\s*>",
    )
)

# A flagged sentence continues onto following lines until one ends a sentence.
MAX_CONTINUATION_LINES = 3
_SENTENCE_END = re.compile(r"[.!?]['\")\]]*\s*$")


def instruction_lines(texts: list[str]) -> list[int]:
    """1-based numbers of the lines that read like instructions to the model.

    A matching line drags along the lines that continue its sentence (up to
    MAX_CONTINUATION_LINES), since an instruction often wraps.
    """
    found: set[int] = set()
    for i, text in enumerate(texts):
        if not any(p.search(text) for p in INSTRUCTION_PATTERNS):
            continue
        found.add(i + 1)
        j = i
        while (
            not _SENTENCE_END.search(texts[j])
            and j + 1 < len(texts)
            and j - i < MAX_CONTINUATION_LINES
        ):
            j += 1
            found.add(j + 1)
    return sorted(found)


def _injection_flag(texts: list[str]) -> Flag | None:
    numbers = instruction_lines(texts)
    if not numbers:
        return None
    quoted = "; ".join(f"L{n}: {texts[n - 1][:80]!r}" for n in numbers)
    return Flag(
        code=FlagCode.PROMPT_INJECTION_SUSPECTED,
        message=f"lines withheld from the model as possible instructions: {quoted}",
    )


def _lines_as_sent(ocr: OcrResult) -> list[str]:
    """The OCR lines the model sees, with instruction-like lines replaced by WITHHELD.

    Each field's raw_text is checked against these, so a value quoted from a
    withheld line is rejected too.
    """
    texts = _ocr_lines(ocr)
    withheld = set(instruction_lines(texts))
    return [
        WITHHELD if n in withheld else text for n, text in enumerate(texts, start=1)
    ]


def format_user_message(
    ocr: OcrResult, declared_type: DocumentType, country: str
) -> str:
    """The request in the format prompts/system_prompt.md describes.

    Lines that read like instructions to the model are replaced by WITHHELD.
    """
    numbered = "\n".join(
        f"L{n}: {text}" for n, text in enumerate(_lines_as_sent(ocr), start=1)
    )
    parts = [
        f"<declared_type>{declared_type}</declared_type>",
        f"<country>{country}</country>",
        f"<ocr_markdown>\n{numbered}\n</ocr_markdown>",
    ]
    if ocr.engine == "pp-ocr":
        scores = [
            {"line": n, "confidence": round(line.confidence, 2)}
            for n, line in enumerate(ocr.lines, start=1)
            if line.confidence is not None
        ]
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
    """Extract one document of `declared_type` from OCR text. Doesn't raise on model failure.

    Every value's raw_text must be found in the lines the model was given (see
    schemas.source_lines). The check runs inside Instructor's retry loop, so a miss
    goes back to the model, and again in _salvage.
    """
    settings = settings or get_settings()
    with source_lines(_lines_as_sent(ocr)):
        return _extract(ocr, declared_type, country, settings)


def _extract(
    ocr: OcrResult, declared_type: DocumentType, country: str, settings: Settings
) -> KYCExtraction:
    model_cls = DOCUMENT_MODELS[declared_type]
    injection = _injection_flag(_ocr_lines(ocr))
    client = instructor.from_openai(_openai_client(settings), mode=instructor.Mode.JSON)

    try:
        doc = client.chat.completions.create(
            model=settings.structuring_model,
            response_model=model_cls,
            max_retries=MAX_RETRIES,
            reasoning_effort=REASONING_EFFORT,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_user_message(ocr, declared_type, country),
                },
            ],
        )
    except MODEL_FAILURES as exc:
        salvaged = _salvage(exc, model_cls)
        if salvaged is None:
            reason = f"extraction failed: {_short_error(exc)}"
            logger.warning("%s %s", declared_type, reason)
            return _failed(declared_type, country, reason, injection)
        data, rejected = salvaged
        doc = model_cls.model_validate(data)
        flags = []
        # Nothing rejected means earlier attempts covered every field the last one
        # got wrong; each such field's note says so.
        if rejected:
            logger.warning(
                "%s extraction incomplete, rejected: %s",
                declared_type,
                ", ".join(rejected),
            )
            flags.append(
                Flag(
                    code=FlagCode.EXTRACTION_INCOMPLETE,
                    message=f"after {MAX_RETRIES + 1} attempts these fields were "
                    "still invalid and were set to null: "
                    + "; ".join(f"{name} ({why})" for name, why in rejected.items()),
                )
            )
    else:
        # Rebuild from the fields alone, so the only flags are the ones code sets.
        data = doc.model_dump(exclude={"flags"})
        flags = []

    flags += [f for f in (_type_mismatch(doc, declared_type), injection) if f]
    return _build(declared_type, country, data, flags)


# --- keeping what was valid when the model never fully succeeds ----------------
#
# Seen live: on a national ID with a two-digit-year DOB, qwen3:4b expanded the
# year twice (rejected), then fixed it but garbled the issue date's raw_text
# (rejected). Throwing the whole document away lost a correct name, number and
# expiry. So on final failure the answers are kept field by field.
#
# Retries can also break a field that was right before: live, on the same ID, the
# expiry was valid ("14.01.2032") on the first attempt and garbled ("14.01.2 2032")
# on the last. So a field the last attempt got wrong falls back to the latest
# earlier attempt where it's valid.

MAX_REASON_CHARS = 200


def _salvage(
    exc: Exception, model_cls: type[BaseDocument]
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """The valid fields from the model's answers, plus {rejected field: why}.

    Each field comes from the last answer if it's valid there, otherwise from the
    latest earlier answer where it is; its note then says which attempt. A field
    the last answer validly left null stays null. A field valid in no answer is
    rejected, with the last answer's reason and raw_text.

    None when there's nothing to keep: no answer (Ollama down), no answer that is a
    JSON object, or every field rejected.

    Fields are added to an all-null document one at a time and kept only if the
    document still validates, so document-level checks (the DOB age range) reject
    the field that breaks them.
    """
    answers = _answers(exc)
    if not answers:
        return None
    last_number, last = answers[-1]

    data: dict[str, Any] = {name: _null_field() for name in field_names(model_cls)}
    rejected: dict[str, str] = {}
    for name in field_names(model_cls):
        why = "missing from the model's answer"
        for number, answer in reversed(answers):
            candidate = answer.get(name)
            if candidate is None:
                continue
            if number != last_number and isinstance(candidate, dict):
                candidate = _from_earlier(candidate, number, why)
            try:
                model_cls.model_validate({**data, name: candidate})
            except ValidationError as err:
                if number == last_number:
                    why = _clip(err.errors()[0]["msg"])
                continue
            data[name] = candidate
            break
        else:
            rejected[name] = why

    if len(rejected) == len(data):
        return None
    for name, why in rejected.items():
        raw = last.get(name)
        raw_text = raw.get("raw_text") if isinstance(raw, dict) else None
        data[name] = _null_field(
            raw_text if isinstance(raw_text, str) else None,
            note=f"rejected by validation: {why}",
        )
    return data, rejected


def _from_earlier(candidate: dict[str, Any], number: int, why: str) -> dict[str, Any]:
    """An earlier attempt's field, its note saying where it came from and why."""
    origin = (
        f"taken from attempt {number}; the latest answer's value was rejected: {why}"
    )
    note = candidate.get("note")
    return {**candidate, "note": f"{note}; {origin}" if note else origin}


def _answers(exc: Exception) -> list[tuple[int, dict[str, Any]]]:
    """(attempt number, JSON object) for each attempt that gave one, oldest first.

    Instructor's retry exception keeps every attempt in failed_attempts; other
    failures (cut-off output) have only last_completion.
    """
    attempts = getattr(exc, "failed_attempts", None) or []
    completions = [getattr(a, "completion", None) for a in attempts]
    if not completions:
        completions = [getattr(exc, "last_completion", None)]
    answers = []
    for number, completion in enumerate(completions, start=1):
        answer = _answer(completion)
        if answer is not None:
            answers.append((number, answer))
    return answers


def _answer(completion: Any) -> dict[str, Any] | None:
    """The JSON object in one completion, if there is one."""
    try:
        content = completion.choices[0].message.content  # pyright: ignore[reportOptionalMemberAccess]
    except (AttributeError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    # Instructor's JSON mode tolerates fences and chatter around the object; so do we.
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        answer = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def _null_field(raw_text: str | None = None, note: str | None = None) -> dict[str, Any]:
    return {"value": None, "raw_text": raw_text, "confidence": "none", "note": note}


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1] + "…"


def _short_error(exc: Exception) -> str:
    """One readable line. A retry exception's str() is a dump of every attempt."""
    if isinstance(exc, InstructorRetryException):
        return f"no valid answer after {MAX_RETRIES + 1} attempts"
    return _clip(f"{type(exc).__name__}: {exc}")


def _type_mismatch(doc: BaseDocument, declared_type: DocumentType) -> Flag | None:
    """DOCUMENT_TYPE_MISMATCH when none of the type's identifying fields has a value."""
    names = IDENTIFYING_FIELDS[declared_type]
    if any(getattr(doc, name).value is not None for name in names):
        return None
    return Flag(
        code=FlagCode.DOCUMENT_TYPE_MISMATCH,
        message=f"no {' or '.join(names)} found; the document may not be a "
        f"{declared_type}",
    )


def _failed(
    declared_type: DocumentType,
    country: str,
    reason: str,
    injection: Flag | None = None,
) -> KYCExtraction:
    """Every field of the declared type at confidence "none", flagged EXTRACTION_FAILED.

    An injection flag is kept too: a reviewer should see the attempt either way.
    """
    data = {
        name: {"value": None, "raw_text": None, "confidence": "none", "note": reason}
        for name in field_names(DOCUMENT_MODELS[declared_type])
    }
    flags = [Flag(code=FlagCode.EXTRACTION_FAILED, message=reason)]
    if injection is not None:
        flags.append(injection)
    return _build(declared_type, country, data, flags)


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
