"""Tests for the KYC extraction schemas in api/schemas.py.

Errors are for structurally wrong data; flags are for things a reviewer should see.
"""

from datetime import date
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from api import schemas
from api.schemas import (
    Confidence,
    DateField,
    Document,
    ExtractedField,
    FlagCode,
    KYCPacket,
)

TODAY = date(2026, 6, 15)

document_adapter: TypeAdapter[Document] = TypeAdapter(Document)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin 'today' so age and expiry boundaries are deterministic."""
    monkeypatch.setattr(schemas, "today", lambda: TODAY)


# --- helpers ---------------------------------------------------------------


def fld(
    value: Any, confidence: str = "high", raw_text: str | None = None
) -> dict[str, Any]:
    """An extracted-field dict. confidence 'none' forces a null value."""
    if confidence == "none":
        value = None
    return {
        "value": value,
        "raw_text": raw_text
        if raw_text is not None
        else (None if value is None else str(value)),
        "confidence": confidence,
        "note": None,
    }


DOC_FIELDS: dict[str, dict[str, Any]] = {
    "national_id": {
        "date_of_birth": "15/03/1990",
        "expiry_date": "01/01/2030",
        "document_number": "ID123456",
        "issue_date": "01/01/2020",
    },
    "drivers_permit": {
        "date_of_birth": "15/03/1990",
        "expiry_date": "01/01/2030",
        "permit_number": "DP998877",
        "issue_date": "01/01/2020",
        "address": "12 Main Street, Port of Spain",
    },
    "passport": {
        "date_of_birth": "15/03/1990",
        "expiry_date": "01/01/2030",
        "passport_number": "P1234567",
        "nationality": "Trinidad and Tobago",
        "issue_date": "01/01/2020",
    },
    "birth_certificate": {
        "date_of_birth": "15/03/1990",
        "registration_number": "BC-4455",
        "place_of_birth": "San Fernando",
    },
    "utility_bill": {
        "address": "12 Main Street, Port of Spain",
        "provider": "Power Co",
        "statement_date": "01/05/2026",
    },
    "payslip": {
        "employer_name": "Acme Ltd",
        "pay_date": "31/05/2026",
        "address": "12 Main Street, Port of Spain",
    },
    "address_letter": {
        "address": "12 Main Street, Port of Spain",
        "issuer": "First Bank",
        "letter_date": "10/05/2026",
    },
}


def make_doc(
    doc_type: str, name: str | None = "John Smith", **overrides: Any
) -> dict[str, Any]:
    """A minimal valid document dict. Override a field with a raw value or a full field dict."""
    values: dict[str, Any] = {"full_name": name, **DOC_FIELDS[doc_type], **overrides}
    doc: dict[str, Any] = {"document_type": doc_type}
    for key, value in values.items():
        if isinstance(value, dict):
            doc[key] = value
        elif value is None:
            doc[key] = fld(None, "none")
        else:
            doc[key] = fld(value)
    return doc


def packet(*docs: dict[str, Any]) -> KYCPacket:
    return KYCPacket.model_validate({"documents": list(docs)})


def codes(flags: list[schemas.Flag]) -> list[FlagCode]:
    return [f.code for f in flags]


# --- ExtractedField / confidence -----------------------------------------


def test_field_with_high_confidence_and_value() -> None:
    f = ExtractedField[str].model_validate(fld("John Smith"))
    assert f.value == "John Smith"
    assert f.confidence is Confidence.HIGH


def test_field_with_none_confidence_and_null_value() -> None:
    f = ExtractedField[str].model_validate(fld(None, "none"))
    assert f.value is None
    assert f.confidence is Confidence.NONE


def test_none_confidence_with_value_is_an_error() -> None:
    data = {
        "value": "John Smith",
        "raw_text": "John Smith",
        "confidence": "none",
        "note": None,
    }
    with pytest.raises(ValidationError, match="none"):
        ExtractedField[str].model_validate(data)


def test_low_confidence_with_value_is_allowed() -> None:
    f = ExtractedField[str].model_validate(fld("J0hn Sm1th", "low"))
    assert f.confidence is Confidence.LOW


def test_unknown_confidence_is_an_error() -> None:
    with pytest.raises(ValidationError):
        ExtractedField[str].model_validate(fld("x", "certain"))


def test_extra_key_on_field_is_an_error() -> None:
    with pytest.raises(ValidationError):
        ExtractedField[str].model_validate({**fld("x"), "bbox": [0, 0, 1, 1]})


# --- day-first dates -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "05/03/1990",
        "05-03-1990",
        "05.03.1990",
        "5/3/1990",
        "5 March 1990",
        "05 MAR 1990",
        "5 mar 1990",
        "1990-03-05",
        " 05/03/1990 ",
    ],
)
def test_dates_parse_day_first(text: str) -> None:
    assert DateField.model_validate(fld(text)).value == date(1990, 3, 5)


def test_day_above_twelve_parses() -> None:
    assert DateField.model_validate(fld("13/01/1990")).value == date(1990, 1, 13)


@pytest.mark.parametrize(
    "text",
    ["01/13/1990", "31/02/1990", "sometime in 1990", "05/03/90", "", "1990"],
)
def test_unparseable_dates_are_errors(text: str) -> None:
    with pytest.raises(ValidationError):
        DateField.model_validate(fld(text))


def test_date_object_passes_through() -> None:
    assert DateField.model_validate(fld(date(1990, 3, 5))).value == date(1990, 3, 5)


def test_null_date_with_none_confidence() -> None:
    assert DateField.model_validate(fld(None, "none")).value is None


# --- DOB gives age 18-110 --------------------------------------------------


@pytest.mark.parametrize(
    "dob",
    ["15/06/2008", "15/06/1916", "16/06/1915", "15/03/1990"],
    ids=["exactly-18", "exactly-110", "110-day-before-111", "typical"],
)
def test_dob_age_in_range_is_valid(dob: str) -> None:
    doc = document_adapter.validate_python(make_doc("passport", date_of_birth=dob))
    assert isinstance(doc, schemas.Passport)
    assert doc.date_of_birth.value is not None


@pytest.mark.parametrize(
    "dob",
    ["16/06/2008", "15/06/1915", "01/01/2020"],
    ids=["17-turns-18-tomorrow", "exactly-111", "child"],
)
def test_dob_age_out_of_range_is_an_error(dob: str) -> None:
    with pytest.raises(ValidationError, match="age"):
        document_adapter.validate_python(make_doc("passport", date_of_birth=dob))


def test_dob_in_the_future_is_an_error() -> None:
    with pytest.raises(ValidationError, match="age"):
        document_adapter.validate_python(
            make_doc("passport", date_of_birth="01/01/2030")
        )


def test_leap_day_dob_age(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = make_doc("passport", date_of_birth="29/02/2008")
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 2, 28))
    with pytest.raises(ValidationError, match="age"):
        document_adapter.validate_python(doc)
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 3, 1))
    document_adapter.validate_python(doc)


def test_null_dob_skips_age_check() -> None:
    doc = document_adapter.validate_python(make_doc("passport", date_of_birth=None))
    assert isinstance(doc, schemas.Passport)
    assert doc.date_of_birth.value is None


@pytest.mark.parametrize(
    "doc_type", ["national_id", "drivers_permit", "passport", "birth_certificate"]
)
def test_age_check_applies_to_every_dob_document(doc_type: str) -> None:
    with pytest.raises(ValidationError, match="age"):
        document_adapter.validate_python(make_doc(doc_type, date_of_birth="01/01/2015"))


# --- expiry: flag, don't fail ----------------------------------------------


def test_expired_document_is_flagged_not_rejected() -> None:
    doc = document_adapter.validate_python(
        make_doc("passport", expiry_date="01/01/2026")
    )
    assert codes(doc.flags) == [FlagCode.DOCUMENT_EXPIRED]


@pytest.mark.parametrize("expiry", ["15/06/2026", "16/06/2026", "01/01/2030"])
def test_expiry_today_or_later_is_not_flagged(expiry: str) -> None:
    doc = document_adapter.validate_python(make_doc("passport", expiry_date=expiry))
    assert doc.flags == []


def test_null_expiry_is_not_flagged() -> None:
    doc = document_adapter.validate_python(make_doc("national_id", expiry_date=None))
    assert doc.flags == []


@pytest.mark.parametrize("doc_type", ["national_id", "drivers_permit", "passport"])
def test_expiry_check_applies_to_every_expiring_document(doc_type: str) -> None:
    doc = document_adapter.validate_python(make_doc(doc_type, expiry_date="14/06/2026"))
    assert codes(doc.flags) == [FlagCode.DOCUMENT_EXPIRED]


def test_expired_round_trip_does_not_duplicate_flag() -> None:
    doc = document_adapter.validate_python(
        make_doc("passport", expiry_date="01/01/2026")
    )
    again = document_adapter.validate_python(doc.model_dump(mode="json"))
    assert codes(again.flags) == [FlagCode.DOCUMENT_EXPIRED]


# --- document types / discriminator ----------------------------------------


@pytest.mark.parametrize("doc_type", list(DOC_FIELDS))
def test_every_document_type_builds(doc_type: str) -> None:
    doc = document_adapter.validate_python(make_doc(doc_type))
    assert doc.document_type == doc_type
    assert doc.full_name.value == "John Smith"


def test_document_types_cover_the_spec() -> None:
    assert set(DOC_FIELDS) == {
        "national_id",
        "drivers_permit",
        "passport",
        "birth_certificate",
        "utility_bill",
        "payslip",
        "address_letter",
    }


def test_unknown_document_type_is_an_error() -> None:
    with pytest.raises(ValidationError):
        document_adapter.validate_python(
            {**make_doc("passport"), "document_type": "bank_statement"}
        )


def test_field_from_another_type_is_an_error() -> None:
    with pytest.raises(ValidationError):
        document_adapter.validate_python(
            make_doc("utility_bill", passport_number="P1234567")
        )


def test_missing_field_is_an_error() -> None:
    doc = make_doc("passport")
    del doc["passport_number"]
    with pytest.raises(ValidationError):
        document_adapter.validate_python(doc)


# --- KYCPacket cross-document checks ---------------------------------------


def test_matching_documents_have_no_flags() -> None:
    p = packet(make_doc("passport"), make_doc("national_id"))
    assert p.flags == []


@pytest.mark.parametrize(
    "other", ["SMITH, JOHN", "john smith", "John  Smith.", "JOHN SMITH"]
)
def test_name_order_case_and_punctuation_are_ignored(other: str) -> None:
    p = packet(make_doc("passport"), make_doc("national_id", name=other))
    assert p.flags == []


def test_small_name_typo_still_matches() -> None:
    p = packet(
        make_doc("passport", name="Jonathan Smith"),
        make_doc("national_id", name="Jonathon Smith"),
    )
    assert p.flags == []


def test_different_name_is_flagged_not_rejected() -> None:
    p = packet(make_doc("passport"), make_doc("national_id", name="Mary Jones"))
    assert codes(p.flags) == [FlagCode.NAME_MISMATCH]
    assert p.flags[0].documents == [0, 1]


def test_name_threshold_is_85() -> None:
    assert schemas.NAME_MATCH_THRESHOLD == 85


def test_different_dob_is_flagged_not_rejected() -> None:
    p = packet(
        make_doc("passport"), make_doc("birth_certificate", date_of_birth="16/03/1990")
    )
    assert codes(p.flags) == [FlagCode.DOB_MISMATCH]
    assert p.flags[0].documents == [0, 1]


def test_dob_match_ignores_date_format() -> None:
    p = packet(
        make_doc("passport"),
        make_doc("birth_certificate", date_of_birth="15 March 1990"),
    )
    assert p.flags == []


def test_document_without_dob_is_only_name_checked() -> None:
    p = packet(make_doc("passport"), make_doc("utility_bill", name="Mary Jones"))
    assert codes(p.flags) == [FlagCode.NAME_MISMATCH]


def test_null_name_is_skipped() -> None:
    p = packet(make_doc("passport"), make_doc("utility_bill", name=None))
    assert p.flags == []


def test_null_dob_is_skipped() -> None:
    p = packet(make_doc("passport"), make_doc("national_id", date_of_birth=None))
    assert p.flags == []


def test_low_confidence_mismatch_is_flagged() -> None:
    p = packet(
        make_doc("passport"),
        make_doc("national_id", full_name=fld("Mary Jones", "low")),
    )
    assert codes(p.flags) == [FlagCode.NAME_MISMATCH]


def test_odd_one_out_is_flagged_against_each_other_document() -> None:
    p = packet(
        make_doc("passport"),
        make_doc("national_id"),
        make_doc("payslip", name="Mary Jones"),
    )
    assert codes(p.flags) == [FlagCode.NAME_MISMATCH, FlagCode.NAME_MISMATCH]
    assert [f.documents for f in p.flags] == [[0, 2], [1, 2]]


def test_single_document_packet_has_no_cross_flags() -> None:
    assert packet(make_doc("passport")).flags == []


def test_empty_packet_is_an_error() -> None:
    with pytest.raises(ValidationError):
        packet()


def test_packet_round_trip_does_not_duplicate_flags() -> None:
    p = packet(
        make_doc("passport"),
        make_doc("national_id", name="Mary Jones", date_of_birth="16/03/1990"),
    )
    again = KYCPacket.model_validate(p.model_dump(mode="json"))
    assert (
        codes(again.flags)
        == codes(p.flags)
        == [FlagCode.NAME_MISMATCH, FlagCode.DOB_MISMATCH]
    )


def test_document_flags_stay_on_the_document() -> None:
    p = packet(make_doc("passport", expiry_date="01/01/2026"), make_doc("national_id"))
    assert p.flags == []
    assert codes(p.documents[0].flags) == [FlagCode.DOCUMENT_EXPIRED]
