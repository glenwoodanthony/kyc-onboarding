"""Tests for scripts/extract_image.py, with OCR and extraction stubbed."""

import json
from pathlib import Path
from typing import Any

import pytest

from api.extract import _failed  # pyright: ignore[reportPrivateUsage]
from api.ocr import Line, OcrResult
from api.schemas import KYCExtraction
from scripts import extract_image

OCR = OcrResult(
    markdown="PASSPORT",
    lines=[Line(text="PASSPORT", confidence=0.97, bbox=(0, 0, 1, 1))],
    engine="pp-ocr",
)

FAILED = _failed("passport", "Fictionland", "extraction failed: Ollama down")


def ok_result() -> KYCExtraction:
    none = {"value": None, "raw_text": None, "confidence": "none", "note": None}
    name = {"value": "ALEX EXAMPLE", "raw_text": "ALEX EXAMPLE", "confidence": "high"}
    return KYCExtraction.model_validate(
        {
            "declared_type": "utility_bill",
            "country": "Fictionland",
            "document": {
                "document_type": "utility_bill",
                "full_name": name,
                "address": none,
                "provider": none,
                "statement_date": none,
            },
        }
    )


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "doc.png"
    path.write_bytes(b"stub")
    return path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub OCR and extraction; record their arguments. Extraction succeeds."""
    seen: dict[str, Any] = {"result": ok_result()}

    def fake_ocr(path: Path, *, use_gpu: bool = True) -> OcrResult:
        seen["ocr"] = (path, use_gpu)
        return OCR

    def fake_extract(ocr: OcrResult, declared_type: str, country: str) -> KYCExtraction:
        seen["extract"] = (ocr, declared_type, country)
        return seen["result"]

    monkeypatch.setattr(extract_image, "ocr_image", fake_ocr)
    monkeypatch.setattr(extract_image, "extract", fake_extract)
    return seen


def test_prints_json_and_ocr_text(
    calls: dict[str, Any], image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = extract_image.main([str(image), "utility_bill", "Fictionland"])

    out, err = capsys.readouterr()
    assert code == 0
    assert json.loads(out) == json.loads(ok_result().model_dump_json())
    assert "L1: PASSPORT  [0.97]" in err
    assert calls["ocr"] == (image, True)
    assert calls["extract"] == (OCR, "utility_bill", "Fictionland")


def test_cpu_flag_skips_gpu(calls: dict[str, Any], image: Path) -> None:
    extract_image.main([str(image), "passport", "Fictionland", "--cpu"])
    assert calls["ocr"] == (image, False)


def test_output_file(
    calls: dict[str, Any],
    image: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_file = tmp_path / "result.json"
    extract_image.main([str(image), "passport", "Fictionland", "-o", str(out_file)])

    assert capsys.readouterr().out == ""
    assert json.loads(out_file.read_text(encoding="utf-8"))["country"] == "Fictionland"


def test_failed_extraction_exits_1_and_shows_flag(
    calls: dict[str, Any], image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls["result"] = FAILED
    code = extract_image.main([str(image), "passport", "Fictionland"])

    assert code == 1
    assert "FLAG EXTRACTION_FAILED: extraction failed: Ollama down" in (
        capsys.readouterr().err
    )


def test_missing_image_exits_2(calls: dict[str, Any], tmp_path: Path) -> None:
    assert extract_image.main([str(tmp_path / "nope.png"), "passport", "X"]) == 2
    assert "ocr" not in calls


def test_unknown_type_is_rejected(image: Path) -> None:
    with pytest.raises(SystemExit):
        extract_image.main([str(image), "library_card", "Fictionland"])
