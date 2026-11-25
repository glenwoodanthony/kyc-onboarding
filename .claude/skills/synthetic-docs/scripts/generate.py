"""Render fictional Caribbean-style KYC documents, plus an answer key.

    uv run python .claude/skills/synthetic-docs/scripts/generate.py [--seed N] [--mismatch] [-o DIR]

Writes seven PNGs and answers.json to synthetic/caribbean/ (or
synthetic/caribbean_mismatch/ with --mismatch). Output is deterministic for a given
seed: Faker and the handwriting jitter are seeded, so rerunning gives byte-identical
files.

Everything is fictional: the island (the Commonwealth of Exampla), its parishes and
towns, the people, the companies and every number. Every image carries a
"SPECIMEN - FICTIONAL" watermark band in its bottom margin, added by save(), the
only way images are written.

answers.json has the same format as synthetic/ids/expected.json, so
scripts/run_synthetic.py --answer-key can score it. Each entry also has
"matches_applicant": whether the document's name should match the national ID's
(KYCPacket raises NAME_MISMATCH when it doesn't).
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from random import Random
from typing import Any

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

SKILL_DIR = Path(__file__).resolve().parent.parent
ROOT = SKILL_DIR.parents[2]
# Running a file puts only its own folder on sys.path; api/ lives at the repo root.
sys.path.insert(0, str(ROOT))

from api.schemas import NAME_MATCH_THRESHOLD, name_similarity

HANDWRITING_FONT = SKILL_DIR / "assets" / "fonts" / "Caveat.ttf"
DEFAULT_OUT = ROOT / "synthetic" / "caribbean"
MISMATCH_OUT = ROOT / "synthetic" / "caribbean_mismatch"
DEFAULT_SEED = 7

WATERMARK = "SPECIMEN - FICTIONAL"
COUNTRY = "Exampla"
COUNTRY_LONG = "Commonwealth of Exampla"

CARD_SIZE = (1000, 640)
PAGE_SIZE = (1000, 1300)

# --- fictional pools ------------------------------------------------------------

FEMALE_NAMES = [
    "Kerry-Ann", "Shanice", "Tamika", "Nadine", "Latoya", "Jhenelle", "Shelly-Ann",
    "Kimone",
]  # fmt: skip
MALE_NAMES = [
    "Dwayne",
    "Andre",
    "Marlon",
    "Kemar",
    "Damion",
    "Rohan",
    "Odane",
    "Garfield",
]
GIVEN_NAMES = FEMALE_NAMES + MALE_NAMES
FEMALE_MIDDLE = ["Maria", "Elizabeth", "Patricia", "Antoinette"]
MALE_MIDDLE = ["Anthony", "Augustus", "Barrington", "Winston"]
ELDER_FEMALE = ["Joyce", "Pearline", "Hyacinth", "Icilda", "Beverley"]
ELDER_MALE = ["Clive", "Desmond", "Lloyd", "Neville", "Egbert"]
SURNAMES = [
    "Campbell", "Thompson", "Williams", "Gordon", "Morrison", "Reid", "Francis",
    "Grant", "McKenzie", "Bailey", "Samuels", "Daley",
]  # fmt: skip
PARISHES = ["St. Example", "St. Sample", "St. Fictive", "St. Placeholder"]
TOWNS = ["Port Fictional", "Sampleton", "Exampleville", "Mockbay", "Demo Heights"]
STREETS = [
    "Hibiscus Lane", "Breadfruit Road", "Sea Grape Avenue", "Poinciana Drive",
    "Ackee Walk",
]  # fmt: skip
EMPLOYERS = [
    "Sampleton Distributors Ltd",
    "Port Fictional Shipping Co. Ltd",
    "Exampla Agro Processors Ltd",
]
POWER_COMPANY = "Exampla Light & Power Co. Ltd"
WATER_COMPANY = "Exampla National Water Commission"


# --- people and dates -----------------------------------------------------------


@dataclass(frozen=True)
class Person:
    given: str
    middle: str
    surname: str

    @property
    def full(self) -> str:
        return f"{self.given} {self.middle} {self.surname}"


def ordinal(day: int) -> str:
    suffix = (
        "th"
        if 11 <= day % 100 <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return f"{day}{suffix}"


def long_date(d: date) -> str:
    """14th March 1988"""
    return f"{ordinal(d.day)} {d:%B %Y}"


def short_date(d: date) -> str:
    """14 MAR 1988"""
    return f"{d.day:02d} {d:%b %Y}".upper()


def numeric_date(d: date) -> str:
    """30/09/2026. Callers pick dates with day > 12, so they can't be read month-first."""
    assert d.day > 12, d
    return f"{d:%d/%m/%Y}"


def add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)


def different_name(
    base: str, make: Callable[[], Person], *, what: str, tries: int = 50
) -> Person:
    """Draw with `make` until the name scores below the NAME_MISMATCH threshold."""
    for _ in range(tries):
        person = make()
        if name_similarity(base, person.full) < NAME_MATCH_THRESHOLD:
            return person
    raise RuntimeError(f"couldn't draw a {what} whose name differs from {base!r}")


# --- rendering -------------------------------------------------------------------


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def hand_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(HANDWRITING_FONT), size)


def render_card(
    title: str,
    issuer: str,
    columns: Sequence[Sequence[tuple[str, str]]],
    *,
    accent: str,
) -> Image.Image:
    """An ID card: coloured header, photo placeholder, label-above-value fields in columns."""
    width, height = CARD_SIZE
    image = Image.new("RGB", CARD_SIZE, "#eef3ee")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (4, 4, width - 5, height - 5), radius=28, outline="#7d8a7f", width=3
    )
    draw.rectangle((7, 24, width - 8, 124), fill=accent)
    draw.text((40, 34), title, fill="white", font=font(40))
    draw.text((40, 86), issuer, fill="white", font=font(26))

    # Photo placeholder: a grey silhouette, no real face.
    draw.rectangle((40, 150, 260, 430), fill="#d3d9d4", outline="#7d8a7f", width=2)
    draw.ellipse((105, 185, 195, 275), fill="#aeb7b0")
    draw.pieslice((65, 290, 235, 470), 180, 360, fill="#aeb7b0")

    for x, fields in zip((290, 660), columns, strict=False):
        y = 150
        for label, value in fields:
            draw.text((x, y), label, fill="#4f5a52", font=font(20))
            draw.text((x, y + 24), value, fill="black", font=font(30))
            y += 74
    return image


def render_page(
    letterhead: str, subtitle: str, lines: Sequence[str], *, accent: str
) -> Image.Image:
    """A letter or statement. "" is a blank line; "left\\tright" makes two columns."""
    width, _ = PAGE_SIZE
    image = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 140), fill=accent)
    draw.text((60, 30), letterhead, fill="white", font=font(44))
    draw.text((60, 92), subtitle, fill="white", font=font(26))

    y = 200
    for line in lines:
        if "\t" in line:
            left, right = line.split("\t")
            draw.text((60, y), left, fill="black", font=font(28))
            draw.text((640, y), right, fill="black", font=font(28))
        elif line:
            draw.text((60, y), line, fill="black", font=font(28))
        y += 44
    return image


def write_by_hand(
    image: Image.Image, xy: tuple[int, int], text: str, max_width: int, rng: Random
) -> None:
    """Paste `text` in handwriting, slightly rotated and offset, shrunk to fit."""
    size = 44
    while size > 24 and hand_font(size).getlength(text) > max_width:
        size -= 2
    f = hand_font(size)
    left, _top, right, bottom = f.getbbox(text)
    layer = Image.new("RGBA", (int(right) + 20, int(bottom) + 20), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((10 - left, 10), text, font=f, fill=(25, 40, 110, 255))
    layer = layer.rotate(
        rng.uniform(-1.5, 1.5), resample=Image.Resampling.BICUBIC, expand=True
    )
    x, y = xy
    image.paste(layer, (x + rng.randint(-4, 4), y - 16 + rng.randint(-3, 3)), layer)


def render_form(
    title: str,
    issuer: str,
    rows: Sequence[tuple[str, str]],
    footer: Sequence[str],
    *,
    accent: str,
    handwritten: bool,
    rng: Random,
) -> Image.Image:
    """A printed register form: labels typed, entries typed or handwritten on rules."""
    width, _ = PAGE_SIZE
    image = Image.new("RGB", PAGE_SIZE, "#fbf8ef")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, width - 21, PAGE_SIZE[1] - 21), outline=accent, width=4)
    draw.text((60, 50), title, fill=accent, font=font(46))
    draw.text((60, 110), issuer, fill="#333333", font=font(26))
    draw.line((60, 160, width - 60, 160), fill=accent, width=2)

    label_x, value_x, y = 60, 340, 220
    for label, value in rows:
        draw.text((label_x, y), label, fill="#333333", font=font(24))
        draw.line((value_x, y + 34, width - 60, y + 34), fill="#9a9a9a", width=1)
        if handwritten:
            write_by_hand(image, (value_x + 6, y), value, width - 70 - value_x, rng)
        else:
            draw.text((value_x + 6, y), value, fill="black", font=font(28))
        y += 80
    y += 20
    for line in footer:
        draw.text((60, y), line, fill="#333333", font=font(24))
        y += 40
    return image


def stamp_specimen(image: Image.Image) -> Image.Image:
    """Mark the image "SPECIMEN - FICTIONAL" in a band in the bottom margin.

    The band is clear of every field, so OCR reads it as one line of its own. Two
    earlier versions put text over the page and broke extraction live:
    - tiled across the whole image (alpha 95): PP-OCR dropped or cut the values
      underneath, leaving 12 of 29 expected values in the OCR text (28 unmarked),
      and every case failed;
    - one large faint diagonal (alpha 50): PP-OCR read it where it crossed blank
      space, and qwen3:4b appended the fragment "FICTION" to the JP letter's address.
    """
    width, height = image.size
    band = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)
    top = height - 78
    draw.rectangle((30, top, width - 31, top + 46), fill=(178, 34, 34, 40))
    text = f"{WATERMARK}  -  NOT A REAL DOCUMENT"
    f = font(28)
    draw.text(
        ((width - draw.textlength(text, font=f)) / 2, top + 8),
        text,
        font=f,
        fill=(178, 34, 34, 255),
    )
    stamped = image.convert("RGBA")
    stamped.alpha_composite(band)
    return stamped.convert("RGB")


# --- the documents ---------------------------------------------------------------


@dataclass
class Document:
    image: str
    declared_type: str
    probes: str
    expected: dict[str, Any]
    matches_applicant: bool
    picture: Image.Image


def build(seed: int = DEFAULT_SEED, *, mismatch: bool = False) -> list[Document]:
    """Draw one fictional applicant and render their seven documents, unwatermarked."""
    fake = Faker()
    fake.seed_instance(seed)
    rng = Random(seed)

    def pick(pool: Sequence[str]) -> str:
        return fake.random_element(pool)

    def between(start: date, end: date) -> date:
        return fake.date_between_dates(date_start=start, date_end=end)

    surname = pick(SURNAMES)
    given = pick(GIVEN_NAMES)
    female = given in FEMALE_NAMES
    applicant = Person(given, pick(FEMALE_MIDDLE if female else MALE_MIDDLE), surname)
    dob = between(date(1965, 1, 1), date(2000, 12, 31))
    parish, town, street = pick(PARISHES), pick(TOWNS), pick(STREETS)
    address_lines = [f"Lot {fake.random_int(2, 98)} {street}", town, parish]
    address = ", ".join(address_lines)

    # The relative on the second bill: an elder, same surname, a different name.
    mother = different_name(
        applicant.full,
        lambda: Person(pick(ELDER_FEMALE), pick(FEMALE_MIDDLE), surname),
        what="relative",
    )
    father = Person(pick(ELDER_MALE), pick(MALE_MIDDLE), surname)
    # --mismatch: the birth certificate carries a different (birth) surname.
    # Drawn even without --mismatch, so both runs make the same later draws and
    # differ only in the birth certificates' surnames.
    birth_name = different_name(
        applicant.full,
        lambda: Person(
            applicant.given,
            applicant.middle,
            pick([s for s in SURNAMES if s != surname]),
        ),
        what="birth name",
    )
    at_birth = applicant
    if mismatch:
        at_birth = birth_name
        father = Person(father.given, father.middle, birth_name.surname)

    docs: list[Document] = []

    # National ID: surname and given names on separate lines, as on many cards.
    id_number = fake.bothify("###-###-###")
    issued = between(date(2020, 1, 1), date(2024, 12, 31))
    expires = add_years(issued, 10)
    docs.append(
        Document(
            "national_id.png",
            "national_id",
            "Typed card; surname and given names printed separately, so full_name "
            "must be assembled; dates like '14 MAR 1988'.",
            {
                "full_name": applicant.full.upper(),
                "date_of_birth": dob.isoformat(),
                "expiry_date": expires.isoformat(),
                "document_number": id_number,
                "issue_date": issued.isoformat(),
            },
            True,
            render_card(
                "NATIONAL IDENTIFICATION CARD",
                f"Electoral & Registration Office, {COUNTRY_LONG}",
                [
                    [
                        ("Surname", surname.upper()),
                        (
                            "Given Names",
                            f"{applicant.given} {applicant.middle}".upper(),
                        ),
                        ("Date of Birth", short_date(dob)),
                        ("Parish", parish.upper()),
                    ],
                    [
                        ("ID No.", id_number),
                        ("Date of Issue", short_date(issued)),
                        ("Date of Expiry", short_date(expires)),
                    ],
                ],
                accent="#1d6b45",
            ),
        )
    )

    # Birth certificate, typed and handwritten: same register entry.
    registration = f"{fake.bothify('B-#####')}/{dob.year}"
    birth_place = f"{pick(TOWNS)} Public Hospital, {pick(PARISHES)}"
    registered = dob + timedelta(days=fake.random_int(7, 60))
    cert_rows = [
        ("Registration No.", registration),
        ("Name of Child", at_birth.full),
        ("Sex", "Female" if female else "Male"),
        ("Date of Birth", long_date(dob)),
        ("Place of Birth", birth_place),
        ("Name of Mother", mother.full),
        ("Name of Father", father.full),
        ("Date of Registration", long_date(registered)),
    ]
    cert_expected = {
        "full_name": at_birth.full,
        "date_of_birth": dob.isoformat(),
        "registration_number": registration,
        "place_of_birth": birth_place,
    }
    mismatch_note = (
        " The child's surname differs from the ID (--mismatch), so a packet with "
        "the ID gets NAME_MISMATCH."
        if mismatch
        else ""
    )
    for handwritten in (False, True):
        style = "handwritten" if handwritten else "typed"
        docs.append(
            Document(
                f"birth_certificate_{style}.png",
                "birth_certificate",
                f"Registrar General's form with {style} entries; the mother's and "
                "father's names and the registration date must not be taken for the "
                f"child's.{mismatch_note}",
                dict(cert_expected),
                not mismatch,
                render_form(
                    "CERTIFICATE OF BIRTH",
                    f"Registrar General's Department, {COUNTRY_LONG}",
                    cert_rows,
                    [
                        "Certified a true copy of an entry in the Register of Births.",
                        f"Deputy Registrar General, {pick(PARISHES)}",
                    ],
                    accent="#6b4a1d",
                    handwritten=handwritten,
                    rng=rng,
                ),
            )
        )

    # Utility bills: one in the applicant's name, one in their mother's.
    for person, company, accent, unit, own in (
        (applicant, POWER_COMPANY, "#c45a00", "kWh", True),
        (mother, WATER_COMPANY, "#1d5b8c", "m3", False),
    ):
        statement = between(date(2026, 8, 1), date(2026, 10, 28))
        used = fake.random_int(80, 640)
        amount = f"EX$ {used * 0.41:,.2f}"
        docs.append(
            Document(
                "utility_bill.png" if own else "utility_bill_relative.png",
                "utility_bill",
                "Bill with a three-line address up to the parish."
                if own
                else "Bill at the applicant's address but in a relative's name: "
                "full_name is the relative's, and a packet with the ID gets "
                "NAME_MISMATCH.",
                {
                    "full_name": person.full.upper(),
                    "address": address,
                    "provider": company,
                    "statement_date": statement.isoformat(),
                },
                own,
                render_page(
                    company.upper(),
                    "Water bill" if unit == "m3" else "Electricity bill",
                    [
                        person.full.upper(),
                        *address_lines,
                        "",
                        f"Bill date: {long_date(statement)}",
                        f"Account no.: {fake.bothify('##-####-####')}",
                        "",
                        f"Consumption ({used} {unit})\t{amount}",
                        f"Amount due\t{amount}",
                        "",
                        "Please pay within 14 days to avoid disconnection.",
                    ],
                    accent=accent,
                ),
            )
        )

    # JP letter: a Justice of the Peace certifies the applicant's address.
    jp = Person(pick(ELDER_MALE), pick(MALE_MIDDLE)[0] + ".", pick(SURNAMES))
    jp_signature = f"{jp.full}, J.P."
    letter_day = between(date(2026, 9, 1), date(2026, 10, 30))
    docs.append(
        Document(
            "jp_letter.png",
            "address_letter",
            "Justice of the Peace letter certifying residence; issuer is the JP's "
            "signature line, and the address is inside the body text.",
            {
                "full_name": applicant.full,
                "address": address,
                "issuer": jp_signature,
                "letter_date": letter_day.isoformat(),
            },
            True,
            render_page(
                "JUSTICE OF THE PEACE",
                f"Parish of {pick(PARISHES)}, {COUNTRY_LONG}",
                [
                    long_date(letter_day),
                    "",
                    "TO WHOM IT MAY CONCERN",
                    "",
                    f"I hereby certify that I have known {applicant.full}",
                    f"for {fake.random_int(4, 15)} years and that they reside at:",
                    "",
                    *address_lines,
                    "",
                    "This letter is issued at the applicant's request for",
                    "proof of address.",
                    "",
                    "Yours faithfully,",
                    "",
                    jp_signature,
                    "Justice of the Peace",
                ],
                accent="#3a3a6b",
            ),
        )
    )

    # Payslip.
    employer = pick(EMPLOYERS)
    pay_day = fake.random_element([date(2026, 9, 30), date(2026, 10, 30)])
    gross = fake.random_int(180, 420) * 1000
    tax = gross * 0.25
    docs.append(
        Document(
            "payslip.png",
            "payslip",
            "Payslip with the address on one line and a numeric pay date.",
            {
                "full_name": applicant.full,
                "employer_name": employer,
                "pay_date": pay_day.isoformat(),
                "address": address,
            },
            True,
            render_page(
                employer,
                f"Payslip - {pay_day:%B %Y}",
                [
                    f"Employee: {applicant.full}",
                    f"Employee No.: {fake.bothify('E####')}",
                    f"Address: {address}",
                    "",
                    f"Pay date: {numeric_date(pay_day)}",
                    "",
                    f"Basic pay\tEX$ {gross:,.2f}",
                    f"Income tax\t- EX$ {tax:,.2f}",
                    f"Net pay\tEX$ {gross - tax:,.2f}",
                ],
                accent="#123d5c",
            ),
        )
    )
    return docs


def save(image: Image.Image, path: Path) -> None:
    """The only way images are written, so every one carries the watermark."""
    stamp_specimen(image).save(path)


def key_path(path: Path) -> str:
    """Repo-relative when inside the repo, as run_synthetic.py expects."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write(docs: Sequence[Document], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = []
    for doc in docs:
        path = out_dir / doc.image
        save(doc.picture, path)
        answers.append(
            {
                "image": key_path(path),
                "declared_type": doc.declared_type,
                "country": COUNTRY,
                "probes": doc.probes,
                "expected": doc.expected,
                "flags": [],
                "allowed_flags": [],
                "matches_applicant": doc.matches_applicant,
            }
        )
    key = out_dir / "answers.json"
    key.write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    return key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render fictional Caribbean-style KYC documents and answers.json."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="give the birth certificates a different surname from the ID",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"output folder (default {DEFAULT_OUT.relative_to(ROOT).as_posix()}, "
        f"or {MISMATCH_OUT.relative_to(ROOT).as_posix()} with --mismatch)",
    )
    args = parser.parse_args(argv)
    out_dir = args.output or (MISMATCH_OUT if args.mismatch else DEFAULT_OUT)
    docs = build(args.seed, mismatch=args.mismatch)
    key = write(docs, out_dir)
    for doc in docs:
        print(f"wrote {out_dir / doc.image}")
    print(f"wrote {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
