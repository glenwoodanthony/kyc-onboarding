"""Render synthetic/ocr_sample.png: a fictional passport-style card for OCR tests.

Run with `uv run python synthetic/make_ocr_sample.py`. All text is fictional.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LINES = [
    "PASSPORT",
    "Republic of Fictionland",
    "Surname: EXAMPLE",
    "Given names: ALEX JORDAN",
    "Date of birth: 04/07/1990",
    "Passport No.: FX1234567",
]

OUT = Path(__file__).parent / "ocr_sample.png"


def main() -> None:
    font = ImageFont.load_default(size=36)
    image = Image.new("RGB", (900, 60 + 60 * len(LINES)), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(LINES):
        draw.text((40, 30 + 60 * i), line, fill="black", font=font)
    image.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
