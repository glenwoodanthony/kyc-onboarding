"""Tests for scripts/run_synthetic.py, with OCR and extraction stubbed."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from api import schemas
from api.ocr import Line, OcrResult
from api.schemas import KYCExtraction
from scripts import run_synthetic

OCR = OcrResult(
    markdown="PAYSLIP",
    lines=[Line(text="PAYSLIP", confidence=0.97, bbox=(0, 0, 1, 1))],
    engine="pp-ocr",
)

NONE = {"value": None, "raw_text": None, "confidence": "none", "note": None}

# Two cases on real synthetic images: the same payslip, read as itself and as a
# passport (as in expected.json, so the case names must still differ).
CASES = [
    {
        "image": "synthetic/ids/payslip.png",
        "declared_type": "payslip",
        "country": "Fictionland",
        "probes": "payslip",
        "expected": {
            "full_name": "Alex Jordan Example",
            "employer_name": "Example Bank",
            "pay_date": "2026-10-31",
            "address": "12 Sample Street, Exampletown FL3 9ZZ",
        },
        "flags": [],
        "allowed_flags": ["EXTRACTION_INCOMPLETE"],
    },
    {
        "image": "synthetic/ids/payslip.png",
        "declared_type": "passport",
        "country": "Fictionland",
        "probes": "wrong type",
        "expected": {"full_name": None, "passport_number": None},
        "flags": ["DOCUMENT_TYPE_MISMATCH"],
        "allowed_flags": [],
    },
]


def value(v: Any, raw: str | None = None) -> dict[str, Any]:
    return {"value": v, "raw_text": raw or v, "confidence": "high", "note": None}


def payslip(**overrides: Any) -> KYCExtraction:
    document = {
        "document_type": "payslip",
        "full_name": value("ALEX JORDAN EXAMPLE"),
        "employer_name": value("Example Bank"),
        "pay_date": value("2026-10-31", "31/10/2026"),
        # Joined with " / " and different spacing: still the same address.
        "address": value("12 Sample Street / Exampletown  FL3 9ZZ"),
    } | overrides
    return KYCExtraction.model_validate(
        {"declared_type": "payslip", "country": "Fictionland", "document": document}
    )


def mismatch() -> KYCExtraction:
    document = {name: NONE for name in schemas.field_names(schemas.Passport)} | {
        "document_type": "passport",
        "flags": [{"code": "DOCUMENT_TYPE_MISMATCH", "message": "no passport_number"}],
    }
    return KYCExtraction.model_validate(
        {"declared_type": "passport", "country": "Fictionland", "document": document}
    )


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 11, 23))


@pytest.fixture
def answer_key(tmp_path: Path) -> Path:
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(CASES), encoding="utf-8")
    return path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub OCR and extraction; record their arguments. Both cases match."""
    seen: dict[str, Any] = {
        "ocr": [],
        "extract": [],
        "results": {"payslip": payslip(), "passport": mismatch()},
    }

    def fake_ocr(path: Path, *, use_gpu: bool = True) -> OcrResult:
        seen["ocr"].append((path, use_gpu))
        return OCR

    def fake_extract(ocr: OcrResult, declared_type: str, country: str) -> KYCExtraction:
        seen["extract"].append((declared_type, country))
        return seen["results"][declared_type]

    monkeypatch.setattr(run_synthetic, "ocr_image", fake_ocr)
    monkeypatch.setattr(run_synthetic, "extract", fake_extract)
    return seen


def run(answer_key: Path, *args: str) -> int:
    return run_synthetic.main(["--answer-key", str(answer_key), *args])


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        (
            "12 Sample Street / Exampletown FL3 9ZZ",
            "12 Sample Street, Exampletown FL3 9ZZ",
        ),
        (
            "12 Sample Street,Exampletown  FL3 9ZZ",
            "12 Sample Street, Exampletown FL3 9ZZ",
        ),
        ("ALEX JORDAN EXAMPLE", "Alex Jordan Example"),
        (date(2026, 10, 31), "2026-10-31"),
        (None, None),
    ],
)
def test_same_value(got: Any, expected: Any) -> None:
    assert run_synthetic.same_value(got, expected)


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        ("ALEXJORDANEXAMPLE", "ALEX JORDAN EXAMPLE"),  # a missing space is wrong
        (
            "12 Sample Street, Exampletown FL39ZZ",
            "12 Sample Street, Exampletown FL3 9ZZ",
        ),
        (date(2026, 3, 4), "2026-04-03"),
        ("2032-01-14", None),
        (None, "2032-01-14"),
    ],
)
def test_different_value(got: Any, expected: Any) -> None:
    assert not run_synthetic.same_value(got, expected)


def test_all_cases_match(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(answer_key) == 0

    out = capsys.readouterr().out
    assert "PASS  00_payslip_as_payslip: 4/4 fields, flags ok" in out
    assert "PASS  01_payslip_as_passport: 2/2 fields, flags ok" in out
    assert "2/2 cases match the answer key" in out
    assert calls["extract"] == [("payslip", "Fictionland"), ("passport", "Fictionland")]
    # PP-OCR on CPU by default.
    assert [use_gpu for _, use_gpu in calls["ocr"]] == [False, False]


def test_gpu_flag(calls: dict[str, Any], answer_key: Path) -> None:
    run(answer_key, "--gpu")
    assert [use_gpu for _, use_gpu in calls["ocr"]] == [True, True]


def test_wrong_field_fails_and_is_shown(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrong = value("ALEXJORDANEXAMPLE")
    wrong["note"] = "OCR dropped the spaces"
    calls["results"]["payslip"] = payslip(full_name=wrong)

    assert run(answer_key) == 1
    out = capsys.readouterr().out
    assert "FAIL  00_payslip_as_payslip: 3/4 fields" in out
    assert (
        "full_name: got 'ALEXJORDANEXAMPLE' (high), expected 'Alex Jordan Example'"
        in (out)
    )
    assert "note: OCR dropped the spaces" in out
    assert "1/2 cases match the answer key" in out


def test_missing_flag_fails(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unflagged = mismatch()
    unflagged.document.flags = []
    calls["results"]["passport"] = unflagged

    assert run(answer_key) == 1
    out = capsys.readouterr().out
    assert "FAIL  01_payslip_as_passport: 2/2 fields, flags wrong" in out
    assert "flags: got [], expected ['DOCUMENT_TYPE_MISMATCH']" in out


INCOMPLETE = {"code": "EXTRACTION_INCOMPLETE", "message": "address rejected"}


def test_allowed_flag_passes(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls["results"]["payslip"] = payslip(flags=[INCOMPLETE])

    assert run(answer_key) == 0
    assert (
        "PASS  00_payslip_as_payslip: 4/4 fields, flags ok" in capsys.readouterr().out
    )


def test_flag_neither_expected_nor_allowed_fails(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    extra = mismatch()
    extra.document.flags.append(schemas.Flag.model_validate(INCOMPLETE))
    calls["results"]["passport"] = extra

    assert run(answer_key) == 1
    out = capsys.readouterr().out
    assert "FAIL  01_payslip_as_passport: 2/2 fields, flags wrong" in out
    assert (
        "flags: got ['DOCUMENT_TYPE_MISMATCH', 'EXTRACTION_INCOMPLETE'], "
        "expected ['DOCUMENT_TYPE_MISMATCH']" in out
    )


def test_allowed_flags_are_shown_when_flags_are_wrong(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls["results"]["payslip"] = payslip(
        flags=[{"code": "DOCUMENT_TYPE_MISMATCH", "message": "no employer_name"}]
    )

    assert run(answer_key) == 1
    assert (
        "flags: got ['DOCUMENT_TYPE_MISMATCH'], expected [], "
        "allowed ['EXTRACTION_INCOMPLETE']" in capsys.readouterr().out
    )


def test_only_filters_cases(
    calls: dict[str, Any], answer_key: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(answer_key, "--only", "as_passport") == 0
    assert calls["extract"] == [("passport", "Fictionland")]
    assert "1/1 cases match" in capsys.readouterr().out


def test_only_matching_nothing_exits_2(calls: dict[str, Any], answer_key: Path) -> None:
    assert run(answer_key, "--only", "library_card") == 2
    assert calls["ocr"] == []


def test_output_dir(calls: dict[str, Any], answer_key: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    run(answer_key, "-o", str(out_dir))

    extraction = json.loads((out_dir / "00_payslip_as_payslip.json").read_text("utf-8"))
    assert extraction["declared_type"] == "payslip"
    ocr_text = (out_dir / "01_payslip_as_passport.ocr.txt").read_text("utf-8")
    assert ocr_text == "engine: pp-ocr\nL1: PAYSLIP  [0.97]\n"
    summary = json.loads((out_dir / "summary.json").read_text("utf-8"))
    assert [(case["case"], case["ok"]) for case in summary] == [
        ("00_payslip_as_payslip", True),
        ("01_payslip_as_passport", True),
    ]


def test_missing_answer_key_exits_2(calls: dict[str, Any], tmp_path: Path) -> None:
    assert run(tmp_path / "nope.json") == 2
    assert calls["ocr"] == []


def test_missing_image_exits_2(
    calls: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = tmp_path / "expected.json"
    key.write_text(
        json.dumps([CASES[0] | {"image": "synthetic/ids/nope.png"}]), encoding="utf-8"
    )
    assert run(key) == 2
    assert "synthetic/ids/nope.png" in capsys.readouterr().err
    assert calls["ocr"] == []


def test_real_answer_key_case_names_are_distinct() -> None:
    # The payslip image is run twice, so names must include the declared type.
    cases = json.loads(run_synthetic.ANSWER_KEY.read_text(encoding="utf-8"))
    names = [run_synthetic.case_name(i, c) for i, c in enumerate(cases)]
    assert len(set(names)) == len(names) == len(cases)
