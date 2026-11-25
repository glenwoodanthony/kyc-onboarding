"""Pydantic models for KYC document extraction.

Every extracted value is wrapped in an ExtractedField carrying its raw text, the
extractor's confidence and a note.

Rule of thumb: data that is structurally wrong raises a ValidationError (unparseable
date, impossible age, a value claimed with confidence "none"). Things a human reviewer
should look at become Flags (expired document, names or DOBs that disagree across
documents) and never stop the data from loading.
"""

import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
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
    # Set by api.extract: the last attempt still had invalid fields; those were set
    # to null and the rest kept. The message names them and why.
    EXTRACTION_INCOMPLETE = "EXTRACTION_INCOMPLETE"
    # Set by api.extract: none of the declared type's identifying fields (its number
    # and defining date) was found, so the text may be another document.
    DOCUMENT_TYPE_MISMATCH = "DOCUMENT_TYPE_MISMATCH"
    # Set by api.extract: some document lines read like instructions to the model,
    # so they were withheld from it. A reviewer should look at the original.
    PROMPT_INJECTION_SUSPECTED = "PROMPT_INJECTION_SUSPECTED"


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
    # "4th July, 1990" -> "4 July 1990": drop ordinal suffixes and commas.
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.replace(",", " ").split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"cannot parse {text!r} as a day-first date")


def date_in_raw_text(raw_text: str) -> date | None:
    """The day-first date raw_text gives, or None if it doesn't give a full one.

    Accepts a leading label ("Date of birth: 04/07/1990") as well as the bare date.
    """
    for candidate in (raw_text, raw_text.rpartition(":")[2]):
        try:
            return parse_day_first(candidate)
        except ValueError:
            continue
    return None


class DateField(ExtractedField[date]):
    @field_validator("value", mode="before")
    @classmethod
    def _parse_day_first(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return parse_day_first(value)
        return value

    @model_validator(mode="after")
    def _value_follows_from_raw_text(self) -> Self:
        # Catches a model "fixing" what the document doesn't say: expanding a
        # two-digit year (04.07.90 -> 1990-07-04) or padding a bare year
        # (2030 -> 2030-01-01). Applies at every confidence level. The messages are
        # sent back to the model on retry, so they say what to do.
        if self.value is None:
            return self
        if self.raw_text is None:
            raise ValueError(
                f"date value {self.value} has no raw_text; copy the date as printed "
                "into raw_text, or set value to null"
            )
        read = date_in_raw_text(self.raw_text)
        if read is None:
            raise ValueError(
                f"raw_text {self.raw_text!r} is not a full day-first date with a "
                "four-digit year, so it can't support a value; set value to null, "
                'confidence to "low" or "none", and explain in note'
            )
        if read != self.value:
            raise ValueError(
                f"value {self.value} does not match raw_text {self.raw_text!r}, "
                f"which reads day-first as {read}"
            )
        return self


def _without_whitespace(text: str) -> str:
    return "".join(text.split())


def _value_is_in_raw_text(field: ExtractedField[str]) -> ExtractedField[str]:
    # A number must be copied, not retyped: its characters must appear, in order
    # and unchanged, in raw_text. Whitespace is ignored, so "FX 765 4321" can give
    # "FX7654321", and a leading label is allowed. Applies at every confidence
    # level. Live, qwen3:4b gave "FL-4821-773:0" for raw_text "FL-4821-7730", which
    # the raw_text-in-source check can't catch: it compares only letters and digits.
    if field.value is None:
        return field
    if field.raw_text is None:
        raise ValueError(
            f"number {field.value!r} has no raw_text; copy the number as printed "
            "into raw_text, or set value to null"
        )
    if _without_whitespace(field.value) not in _without_whitespace(field.raw_text):
        raise ValueError(
            f"value {field.value!r} is not in raw_text {field.raw_text!r}; copy "
            "the number into value exactly as it appears in raw_text, with no "
            "added, removed or changed characters, or set value to null"
        )
    return field


IDENTIFIER_CHECK = AfterValidator(_value_is_in_raw_text)

# A document, permit, passport or registration number. It's an Annotated
# ExtractedField[str], not a subclass, so the JSON schema sent to the model is
# unchanged: a subclass adds its own $defs entry, about 150 tokens on a request
# already at Ollama's default 4,096-token context. Live, that extra length changed
# qwen3:4b's answers and sent it into a repetition loop that Ollama aborted.
IdentifierField = Annotated[ExtractedField[str], IDENTIFIER_CHECK]


def age_on(born: date, on: date) -> int:
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


# --- documents -------------------------------------------------------------


# The source text a document is being extracted from: the OCR lines exactly as the
# model saw them. While set (by `source_lines`), every field's raw_text must be found
# in these lines. Unset, as when loading stored results, nothing is checked.
#
# A ContextVar rather than Pydantic's validation context, because the check has to
# run inside Instructor's retry loop, and Instructor's `context=` also renders the
# messages as Jinja templates. The OCR text is written by the customer, so it must
# never be rendered.
_SOURCE_LINES: ContextVar[tuple[str, ...] | None] = ContextVar(
    "source_lines", default=None
)

# raw_text joins the lines of a multi-line value with " / " (prompt rule 4).
_RAW_TEXT_JOIN = re.compile(r"\s+/\s+")


@contextmanager
def source_lines(lines: Iterable[str]) -> Iterator[None]:
    """Within this block, every raw_text must appear in `lines`."""
    token = _SOURCE_LINES.set(tuple(lines))
    try:
        yield
    finally:
        _SOURCE_LINES.reset(token)


def _letters_and_digits(text: str) -> str:
    # OCR drops and adds spaces (Name:ALEXJORDANEXAMPLE), and the model may repair
    # them, so only letters and digits are compared, without case.
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def raw_text_in_source(raw_text: str, lines: Sequence[str]) -> bool:
    """Whether every " / "-separated part of raw_text is in one line, or two in a row.

    Two in a row allows a label and its value on consecutive lines ("Date of birth"
    then "04.07.1990") quoted as one. Letters and digits must match exactly, so a
    date that is not on the document ("15.07.2022" for "15.01.2022") is not found.
    """
    normalised = [_letters_and_digits(line) for line in lines]
    windows = normalised + [a + b for a, b in pairwise(normalised)]
    for part in _RAW_TEXT_JOIN.split(raw_text):
        needle = _letters_and_digits(part)
        if needle and not any(needle in window for window in windows):
            return False
    return True


def identifier_in_source(value: str, lines: Sequence[str]) -> bool:
    """Whether a number is in one line, or two in a row, with its punctuation.

    Only whitespace is ignored, so OCR's dropped or added spaces don't matter, but
    an added, removed or changed "-", "/" or "." does. raw_text_in_source can't
    catch that: it compares only letters and digits.
    """
    needle = _without_whitespace(value)
    stripped = [_without_whitespace(line) for line in lines]
    windows = stripped + [a + b for a, b in pairwise(stripped)]
    return any(needle in window for window in windows)


class BaseDocument(StrictModel):
    # Every document type names its holder `full_name` so packet checks are uniform.
    full_name: ExtractedField[str]
    flags: list[Flag] = []

    @model_validator(mode="after")
    def _raw_text_is_in_source(self) -> Self:
        # Catches a value invented together with a raw_text that agrees with it, which
        # the date check can't: live, qwen3:4b gave issue_date 2022-07-15 with
        # raw_text "15.07.2022" for a card printed 15.01.2022. The message goes back
        # to the model on retry.
        lines = _SOURCE_LINES.get()
        if lines is None:
            return self
        problems = []
        for name, info in type(self).model_fields.items():
            field = getattr(self, name)
            if not isinstance(field, ExtractedField) or field.value is None:
                continue
            is_number = IDENTIFIER_CHECK in info.metadata
            if field.raw_text is None:
                problems.append(
                    f"{name} has a value but no raw_text; copy the text it comes "
                    "from into raw_text, or set value to null"
                )
            elif not raw_text_in_source(field.raw_text, lines):
                problems.append(
                    f"{name}.raw_text {field.raw_text!r} does not appear in the OCR "
                    "text; copy raw_text exactly from the document's lines and take "
                    "value from that same text, or set value and raw_text to null"
                )
            elif is_number and not identifier_in_source(str(field.value), lines):
                # Live, qwen3:4b gave "B-13909/1.968" in both value and raw_text for
                # a certificate OCR read as "B-13909/1968". The value matches its
                # raw_text, and the raw_text matches the source on letters and digits.
                problems.append(
                    f"{name} {field.value!r} does not appear in the OCR text with "
                    "those characters; copy the number exactly as the OCR text shows "
                    "it, with no added, removed or changed punctuation, or set value "
                    "and raw_text to null"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return self


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
    document_number: IdentifierField
    issue_date: DateField


class DriversPermit(HasDOB, HasExpiry):
    document_type: Literal["drivers_permit"] = "drivers_permit"
    permit_number: IdentifierField
    issue_date: DateField
    address: ExtractedField[str]


class Passport(HasDOB, HasExpiry):
    document_type: Literal["passport"] = "passport"
    passport_number: IdentifierField
    nationality: ExtractedField[str]
    issue_date: DateField


class BirthCertificate(HasDOB):
    document_type: Literal["birth_certificate"] = "birth_certificate"
    registration_number: IdentifierField
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
