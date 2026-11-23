"""Pydantic models for KYC document extraction.

Every extracted value is wrapped in an ExtractedField carrying its raw text, the
extractor's confidence and a note.

Rule of thumb: data that is structurally wrong raises a ValidationError (unparseable
date, impossible age, a value claimed with confidence "none"). Things a human reviewer
should look at become Flags (expired document, names or DOBs that disagree across
documents) and never stop the data from loading.
"""

import re
from datetime import date, datetime
from enum import StrEnum
from itertools import combinations
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rapidfuzz import fuzz

NAME_MATCH_THRESHOLD = 85
MIN_AGE = 18
MAX_AGE = 110

# Tried in order. All are day-first; ISO is year-first and so unambiguous. There is
# deliberately no two-digit-year format.
DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d")


def today() -> date:
    """The current date. Tests monkeypatch this to pin age and expiry boundaries."""
    return date.today()


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class FlagCode(StrEnum):
    DOCUMENT_EXPIRED = "DOCUMENT_EXPIRED"
    NAME_MISMATCH = "NAME_MISMATCH"
    DOB_MISMATCH = "DOB_MISMATCH"
    # Set by api.extract: the model gave no valid document after every attempt.
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    # Set by api.extract: every field came back "none", so the text may not be the
    # declared document type.
    DOCUMENT_TYPE_MISMATCH = "DOCUMENT_TYPE_MISMATCH"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Flag(StrictModel):
    code: FlagCode
    message: str
    # Indices into KYCPacket.documents; empty for a flag raised on a single document.
    documents: list[int] = []


class ExtractedField[T](StrictModel):
    value: T | None
    raw_text: str | None = None
    confidence: Confidence
    note: str | None = None

    @model_validator(mode="after")
    def _none_confidence_means_no_value(self) -> Self:
        if self.confidence is Confidence.NONE and self.value is not None:
            raise ValueError("confidence is 'none' but value is not null")
        return self


def parse_day_first(text: str) -> date:
    cleaned = " ".join(text.split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"cannot parse {text!r} as a day-first date")


class DateField(ExtractedField[date]):
    @field_validator("value", mode="before")
    @classmethod
    def _parse_day_first(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return parse_day_first(value)
        return value


def age_on(born: date, on: date) -> int:
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


# --- documents -------------------------------------------------------------


class BaseDocument(StrictModel):
    # Every document type names its holder `full_name` so packet checks are uniform.
    full_name: ExtractedField[str]
    flags: list[Flag] = []


class HasDOB(BaseDocument):
    date_of_birth: DateField

    @model_validator(mode="after")
    def _check_age(self) -> Self:
        dob = self.date_of_birth.value
        if dob is not None:
            age = age_on(dob, today())
            if not MIN_AGE <= age <= MAX_AGE:
                raise ValueError(
                    f"date of birth {dob} gives age {age}, outside {MIN_AGE}-{MAX_AGE}"
                )
        return self


class HasExpiry(BaseDocument):
    expiry_date: DateField

    @model_validator(mode="after")
    def _flag_expired(self) -> Self:
        # Rebuild rather than append, so re-validating a dumped document is idempotent.
        flags = [f for f in self.flags if f.code is not FlagCode.DOCUMENT_EXPIRED]
        expiry = self.expiry_date.value
        if expiry is not None and expiry < today():
            flags.append(
                Flag(
                    code=FlagCode.DOCUMENT_EXPIRED,
                    message=f"document expired on {expiry}",
                )
            )
        self.flags = flags
        return self


class NationalID(HasDOB, HasExpiry):
    document_type: Literal["national_id"] = "national_id"
    document_number: ExtractedField[str]
    issue_date: DateField


class DriversPermit(HasDOB, HasExpiry):
    document_type: Literal["drivers_permit"] = "drivers_permit"
    permit_number: ExtractedField[str]
    issue_date: DateField
    address: ExtractedField[str]


class Passport(HasDOB, HasExpiry):
    document_type: Literal["passport"] = "passport"
    passport_number: ExtractedField[str]
    nationality: ExtractedField[str]
    issue_date: DateField


class BirthCertificate(HasDOB):
    document_type: Literal["birth_certificate"] = "birth_certificate"
    registration_number: ExtractedField[str]
    place_of_birth: ExtractedField[str]


class UtilityBill(BaseDocument):
    document_type: Literal["utility_bill"] = "utility_bill"
    address: ExtractedField[str]
    provider: ExtractedField[str]
    statement_date: DateField


class Payslip(BaseDocument):
    document_type: Literal["payslip"] = "payslip"
    employer_name: ExtractedField[str]
    pay_date: DateField
    address: ExtractedField[str]


class AddressLetter(BaseDocument):
    document_type: Literal["address_letter"] = "address_letter"
    address: ExtractedField[str]
    issuer: ExtractedField[str]
    letter_date: DateField


type Document = Annotated[
    NationalID
    | DriversPermit
    | Passport
    | BirthCertificate
    | UtilityBill
    | Payslip
    | AddressLetter,
    Field(discriminator="document_type"),
]

type DocumentType = Literal[
    "national_id",
    "drivers_permit",
    "passport",
    "birth_certificate",
    "utility_bill",
    "payslip",
    "address_letter",
]

DOCUMENT_MODELS: dict[DocumentType, type[BaseDocument]] = {
    model.model_fields["document_type"].default: model
    for model in get_args(get_args(Document.__value__)[0])
}

# Keys of a document model that are not extracted fields.
NON_FIELD_KEYS = frozenset({"document_type", "flags"})


def field_names(model: type[BaseDocument]) -> list[str]:
    """The extracted-field names of a document model, in declaration order."""
    return [name for name in model.model_fields if name not in NON_FIELD_KEYS]


class KYCExtraction(StrictModel):
    """One document extracted from OCR text, with what the caller declared about it."""

    declared_type: DocumentType
    # Country the document was declared as coming from; a reading hint, not a field.
    country: str
    document: Document


# --- packet ----------------------------------------------------------------


def normalise_name(name: str) -> str:
    """Casefold, drop apostrophes, turn other punctuation into spaces, collapse spaces."""
    name = name.casefold().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^\w\s]", " ", name).split())


def name_similarity(a: str, b: str) -> float:
    """0-100; word order is ignored so 'SMITH, JOHN' matches 'John Smith'."""
    return fuzz.token_sort_ratio(normalise_name(a), normalise_name(b))


class KYCPacket(StrictModel):
    documents: Annotated[list[Document], Field(min_length=1)]
    # Cross-document flags only; each document keeps its own flags.
    flags: list[Flag] = []

    @model_validator(mode="after")
    def _cross_check(self) -> Self:
        # Rebuilt from scratch every validation, so a dump/validate round trip is idempotent.
        flags: list[Flag] = []

        names = [
            (i, d.full_name.value)
            for i, d in enumerate(self.documents)
            if d.full_name.value is not None
        ]
        for (i, a), (j, b) in combinations(names, 2):
            score = name_similarity(a, b)
            if score < NAME_MATCH_THRESHOLD:
                flags.append(
                    Flag(
                        code=FlagCode.NAME_MISMATCH,
                        message=f"name {a!r} vs {b!r} scores {score:.0f} (< {NAME_MATCH_THRESHOLD})",
                        documents=[i, j],
                    )
                )

        dobs = [
            (i, d.date_of_birth.value)
            for i, d in enumerate(self.documents)
            if isinstance(d, HasDOB) and d.date_of_birth.value is not None
        ]
        for (i, a), (j, b) in combinations(dobs, 2):
            if a != b:
                flags.append(
                    Flag(
                        code=FlagCode.DOB_MISMATCH,
                        message=f"date of birth {a} vs {b}",
                        documents=[i, j],
                    )
                )

        self.flags = flags
        return self
