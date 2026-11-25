---
name: synthetic-docs
description: Generate or refresh fictional Caribbean-style KYC test documents (national ID, typed and handwritten birth certificates, utility bills including one in a relative's name, a Justice of the Peace address letter, a payslip) and their answers.json answer key, using Faker and Pillow. Use when asked to make, regenerate or extend these synthetic test images, or to produce a name-mismatch set.
---

# Synthetic Caribbean-style KYC documents

`scripts/generate.py` renders seven fictional documents for one fictional applicant and writes `answers.json`, which holds the correct field values for each image.

```
uv run python .claude/skills/synthetic-docs/scripts/generate.py [--seed N] [--mismatch] [-o DIR]
```

- **Output:** `synthetic/caribbean/` by default, or `synthetic/caribbean_mismatch/` with `--mismatch`. Both sets are committed.
- **Deterministic:** Faker and the handwriting jitter are seeded (default `--seed 7`), so regenerating gives byte-identical files.
- **`--mismatch`:** both birth certificates carry a different surname from the other documents, scoring below the 85 threshold. A `KYCPacket` holding the ID and a birth certificate then gets `NAME_MISMATCH`. Every other value and image is identical to the default set.

## The documents

| Image | `declared_type` | What it probes |
|---|---|---|
| `national_id.png` | national_id | Surname and given names printed separately, so `full_name` must be assembled; dates like `15 APR 1968` |
| `birth_certificate_typed.png` | birth_certificate | Registrar General's form; the parents' names and the registration date must not be taken for the child's |
| `birth_certificate_handwritten.png` | birth_certificate | The same entry, with the values handwritten (Caveat font, seeded tilt and offset) |
| `utility_bill.png` | utility_bill | A three-line address up to the parish |
| `utility_bill_relative.png` | utility_bill | Same address, in a relative's name (same surname, different given names) |
| `jp_letter.png` | address_letter | Justice of the Peace letter certifying residence; `issuer` is the signature line, e.g. `Neville B. Grant, J.P.` |
| `payslip.png` | payslip | Address on one line; a numeric pay date with day > 12 |

Every image carries a "SPECIMEN - FICTIONAL" watermark: a tinted band in the bottom margin reading "SPECIMEN - FICTIONAL - NOT A REAL DOCUMENT", which OCR reads as one line of its own.

## answers.json

It has the same format as `synthetic/ids/expected.json`, so the runner scores it as it is:

```
uv run python scripts/run_synthetic.py --answer-key synthetic/caribbean/answers.json [-o DIR]
```

- **Fields:** each entry has `image`, `declared_type`, `country` (`Exampla`), `probes`, `expected` (dates in ISO format, addresses joined with `, `), `flags` and `allowed_flags`.
- **`matches_applicant`:** each entry also says whether its name should match the ID's. `false` means a packet with the ID raises `NAME_MISMATCH`: always for the relative's bill, and for the birth certificates under `--mismatch`.

## Rules

- **Fictional only.** The island is the Commonwealth of Exampla, and its parishes, towns, companies and people come from the pools at the top of `generate.py` ("St. Example", "Port Fictional", "Exampla Light & Power Co. Ltd"). Never add real place, bank, company or agency names, and never copy anything from `uploads/`.
- **Keep the watermark, and keep it off the values.** Images are written only through `save()`, which calls `stamp_specimen()`. The first version tiled the text across the whole image, and PP-OCR then lost most field values: 12 of 29 expected values were left in the OCR text, against 28 with no watermark, so every document failed. A second version added one faint diagonal; OCR read it where it crossed blank space, and the model appended `FICTION` to the JP letter's address. Keep the watermark in the empty bottom margin, and after changing it check that OCR still reads every value and that no fragment lands next to a field.
- **Output stays under `synthetic/`:** `.gitignore` blocks PNGs anywhere else.
- **Regenerate and commit both sets** after changing the generator, then run `uv run pytest tests/test_synthetic_docs.py`. That test fails if a committed image or answer key drifts from the generator.
- **Dates must be ones `api.schemas.parse_day_first` reads.** That means `15 APR 1968`, `15th April 1968` or `30/09/2026` (numeric only when day > 12), not `15-APR-1968`.
- **Type-checking:** pyright skips files under dot-folders, so it never checks `generate.py`. To check it, copy it to a folder without a leading dot and run `uv run pyright <copy>`. Ruff does lint it.

The handwriting font is Caveat (`assets/fonts/Caveat.ttf`) under the SIL Open Font License, whose text is in `assets/fonts/OFL.txt`.
