"""Tests for api/extract.py. No Ollama needed.

The OpenAI client is real but runs on an httpx2 MockTransport (openai 3.x is built on
httpx2, not httpx) that serves canned chat completions, so Instructor's own parsing
and retry loop are exercised.
"""

import json
from collections.abc import Callable
from datetime import date
from typing import Any, get_args

import httpx2
import openai
import pytest

from api import extract as extract_mod
from api import schemas
from api.extract import MAX_RETRIES, Settings, extract, format_user_message
from api.ocr import Line, OcrResult
from api.schemas import (
    DOCUMENT_MODELS,
    Confidence,
    DocumentType,
    FlagCode,
    KYCExtraction,
    Passport,
    field_names,
)

SETTINGS = Settings(
    ollama_base_url="http://ollama.test/v1",
    structuring_model="test-model",
    _env_file=None,  # pyright: ignore[reportCallIssue]
)

PPOCR = OcrResult(
    markdown="PASSPORT\nEXAMPLE\nALEX JORDAN",
    lines=[
        Line(text="PASSPORT", confidence=0.99, bbox=(0, 0, 10, 10)),
        Line(text="EXAMPLE", confidence=0.95, bbox=(0, 10, 10, 20)),
        Line(text="ALEX JORDAN", confidence=0.642, bbox=(0, 20, 10, 30)),
    ],
    engine="pp-ocr",
)


def field(value: Any, confidence: str = "high") -> dict[str, Any]:
    return {"value": value, "raw_text": value, "confidence": confidence, "note": None}


def passport(**overrides: Any) -> dict[str, Any]:
    doc = {
        "document_type": "passport",
        "full_name": field("ALEX JORDAN EXAMPLE"),
        "date_of_birth": field("1990-07-04"),
        "expiry_date": field("2035-03-11"),
        "passport_number": field("FX12B4567"),
        "nationality": field("FICTIONLAND"),
        "issue_date": field("2025-03-12"),
    }
    return doc | overrides


INVALID = passport(passport_number={"value": "X", "confidence": "none"})


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 6, 15))


Reply = dict[str, Any] | str | httpx2.Response | Exception


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[dict[str, Any]]]:
    """Queue model replies; returns the list of request bodies the model received.

    A reply is a dict (sent as the assistant's content, as JSON), a str (sent as the
    content verbatim), an httpx2.Response (sent as is) or an exception (raised as a
    transport failure). The last reply repeats once the queue runs out.
    """

    def install(*replies: Reply) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(json.loads(request.content))
            reply = replies[min(len(requests), len(replies)) - 1]
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, httpx2.Response):
                return reply
            content = reply if isinstance(reply, str) else json.dumps(reply)
            return httpx2.Response(200, json=completion(content))

        client = openai.OpenAI(
            base_url=SETTINGS.ollama_base_url,
            api_key="test",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
        monkeypatch.setattr(extract_mod, "_openai_client", lambda settings: client)
        return requests

    return install


def completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def run(declared_type: DocumentType = "passport") -> KYCExtraction:
    return extract(PPOCR, declared_type, "Fictionland", settings=SETTINGS)


def assert_failed(result: KYCExtraction, declared_type: DocumentType) -> None:
    doc = result.document
    assert doc.document_type == declared_type
    for name in field_names(DOCUMENT_MODELS[declared_type]):
        f = getattr(doc, name)
        assert (f.value, f.confidence) == (None, Confidence.NONE)
        assert f.note and f.note.startswith("extraction failed")
    assert [f.code for f in doc.flags] == [FlagCode.EXTRACTION_FAILED]


# --- happy path --------------------------------------------------------------


def test_valid_answer_is_returned(model: Callable[..., list[dict[str, Any]]]) -> None:
    requests = model(passport())
    result = run()

    assert len(requests) == 1
    assert result.declared_type == "passport"
    assert result.country == "Fictionland"
    assert isinstance(result.document, Passport)
    assert result.document.date_of_birth.value == date(1990, 7, 4)
    assert result.document.flags == []


def test_request_uses_settings_and_prompt(
    model: Callable[..., list[dict[str, Any]]],
) -> None:
    requests = model(passport())
    run()

    body = requests[0]
    assert body["model"] == "test-model"
    assert body["reasoning_effort"] == "none"  # thinking off
    assert body["temperature"] == 0  # repeatable answers
    system, user = body["messages"][0], body["messages"][-1]
    assert system["role"] == "system"
    assert system["content"].startswith(extract_mod.SYSTEM_PROMPT)
    assert user["content"] == format_user_message(PPOCR, "passport", "Fictionland")


def test_invalid_answer_is_retried_with_the_error(
    model: Callable[..., list[dict[str, Any]]],
) -> None:
    requests = model(INVALID, passport())
    result = run()

    assert len(requests) == 2
    retry_text = json.dumps(requests[1]["messages"])
    assert "confidence is 'none' but value is not null" in retry_text
    assert result.document.flags == []


def test_model_flags_are_dropped_but_expiry_is_still_checked(
    model: Callable[..., list[dict[str, Any]]],
) -> None:
    invented = {"code": "NAME_MISMATCH", "message": "made up", "documents": []}
    model(passport(expiry_date=field("2020-01-01"), flags=[invented]))

    flags = run().document.flags
    assert [f.code for f in flags] == [FlagCode.DOCUMENT_EXPIRED]


def test_all_fields_none_flags_type_mismatch(
    model: Callable[..., list[dict[str, Any]]],
) -> None:
    empty = {"value": None, "raw_text": None, "confidence": "none", "note": "payslip"}
    model(
        {name: empty for name in field_names(Passport)} | {"document_type": "passport"}
    )

    doc = run().document
    assert [f.code for f in doc.flags] == [FlagCode.DOCUMENT_TYPE_MISMATCH]


EMPTY = {"value": None, "raw_text": None, "confidence": "none", "note": None}


def test_name_alone_still_flags_type_mismatch(
    model: Callable[..., list[dict[str, Any]]],
) -> None:
    # Seen live: a payslip declared as passport came back with full_name taken from
    # "Employee: Alex Jordan Example" and every other field null.
    model(
        {name: EMPTY for name in field_names(Passport)}
        | {"document_type": "passport", "full_name": field("Alex Jordan Example")}
    )

    flags = run().document.flags
    assert [f.code for f in flags] == [FlagCode.DOCUMENT_TYPE_MISMATCH]
    assert "no passport_number or date_of_birth found" in flags[0].message


@pytest.mark.parametrize("missing", ["passport_number", "date_of_birth"])
def test_one_identifying_field_is_enough(
    model: Callable[..., list[dict[str, Any]]], missing: str
) -> None:
    # A real passport with one field unreadable is not a type mismatch.
    model(passport(**{missing: EMPTY}))
    assert run().document.flags == []


def test_identifying_fields_cover_every_type() -> None:
    assert set(extract_mod.IDENTIFYING_FIELDS) == set(DOCUMENT_MODELS)
    for doc_type, names in extract_mod.IDENTIFYING_FIELDS.items():
        assert names, doc_type
        assert set(names) <= set(field_names(DOCUMENT_MODELS[doc_type])), doc_type


# --- failure: never raises ----------------------------------------------------


def test_gives_up_after_max_retries(model: Callable[..., list[dict[str, Any]]]) -> None:
    requests = model(INVALID)
    result = run()

    assert MAX_RETRIES == 2
    assert len(requests) == MAX_RETRIES + 1 == 3
    assert_failed(result, "passport")


def test_non_json_answer_falls_back(
    model: Callable[..., list[dict[str, Any]]],
) -> None:
    model("Sorry, I can't help with that.")

    assert_failed(run("utility_bill"), "utility_bill")


def test_ollama_down_falls_back(model: Callable[..., list[dict[str, Any]]]) -> None:
    model(httpx2.ConnectError("connection refused"))

    assert_failed(run("national_id"), "national_id")


def test_unknown_model_falls_back(model: Callable[..., list[dict[str, Any]]]) -> None:
    model(httpx2.Response(404, json={"error": "model 'test-model' not found"}))

    assert_failed(run(), "passport")


@pytest.mark.parametrize("declared_type", get_args(DocumentType.__value__))
def test_fallback_round_trips_with_one_flag(declared_type: DocumentType) -> None:
    failed = extract_mod._failed(declared_type, "Fictionland", "extraction failed: x")  # pyright: ignore[reportPrivateUsage]
    again = KYCExtraction.model_validate(failed.model_dump())

    assert again == failed
    assert_failed(again, declared_type)


# --- request format -------------------------------------------------------------


def test_ppocr_message_numbers_lines_and_gives_confidence() -> None:
    message = format_user_message(PPOCR, "passport", "Fictionland")

    assert message == (
        "<declared_type>passport</declared_type>\n"
        "<country>Fictionland</country>\n"
        "<ocr_markdown>\nL1: PASSPORT\nL2: EXAMPLE\nL3: ALEX JORDAN\n</ocr_markdown>\n"
        "<line_confidence>\n"
        '[{"line": 1, "confidence": 0.99}, {"line": 2, "confidence": 0.95}, '
        '{"line": 3, "confidence": 0.64}]\n'
        "</line_confidence>"
    )


def test_vl_message_skips_blank_lines_and_has_no_confidence() -> None:
    vl = OcrResult(
        markdown="# PASSPORT\n\n**Surname:** EXAMPLE\n   \n",
        lines=[Line(text="PASSPORT", confidence=None, bbox=None)],
        engine="paddleocr-vl",
    )
    message = format_user_message(vl, "passport", "Fictionland")

    assert (
        "<ocr_markdown>\nL1: # PASSPORT\nL2: **Surname:** EXAMPLE\n</ocr_markdown>"
        in (message)
    )
    assert "<line_confidence>" not in message


# --- schema helpers -------------------------------------------------------------


def test_document_models_cover_every_document_type() -> None:
    assert set(DOCUMENT_MODELS) == set(get_args(DocumentType.__value__))
    for doc_type, model_cls in DOCUMENT_MODELS.items():
        assert model_cls.model_fields["document_type"].default == doc_type


def test_settings_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("STRUCTURING_MODEL", raising=False)
    with pytest.raises(ValueError, match="ollama_base_url"):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("STRUCTURING_MODEL", "some-model")
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert (settings.ollama_base_url, settings.structuring_model) == (
        "http://example.test/v1",
        "some-model",
    )
