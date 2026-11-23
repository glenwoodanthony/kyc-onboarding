"""Photo in, validated JSON out: OCR one image and extract it, all on this machine.

    uv run python scripts/extract_image.py IMAGE DECLARED_TYPE COUNTRY [--cpu] [-o FILE]

    uv run python scripts/extract_image.py synthetic/ocr_sample.png passport Fictionland

The JSON goes to stdout (or to FILE with -o). The OCR engine and text go to stderr,
so when a field is wrong you can see whether the OCR misread it or the model did.
Paddle's own log lines go to stderr as well.

Needs the OCR models (`uv sync --group ocr`), Ollama running, and OLLAMA_BASE_URL
and STRUCTURING_MODEL in .env.

Exit code: 0 when a document was extracted, 1 when extraction failed (every field
"none" with an EXTRACTION_FAILED flag). Other flags, such as DOCUMENT_EXPIRED, don't
change the exit code.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import get_args

# Run as a file, Python puts scripts/ on the path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.extract import extract
from api.ocr import ocr_image
from api.schemas import DocumentType, FlagCode

DOCUMENT_TYPES: tuple[str, ...] = get_args(DocumentType.__value__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR one document image and extract it to validated JSON."
    )
    parser.add_argument("image", type=Path, help="the document photo or scan")
    parser.add_argument(
        "declared_type", choices=DOCUMENT_TYPES, help="what the document should be"
    )
    parser.add_argument("country", help="the country said to have issued it")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="skip PaddleOCR-VL and use PP-OCR on CPU: about 1 s a page instead of "
        "40 s (plus a one-off model load), gives per-line confidence, and leaves "
        "the GPU to Ollama",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="write the JSON here instead of stdout"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.image.is_file():
        print(f"error: no such image: {args.image}", file=sys.stderr)
        return 2

    start = time.perf_counter()
    ocr = ocr_image(args.image, use_gpu=not args.cpu)
    ocr_seconds = time.perf_counter() - start
    print(f"\n--- OCR ({ocr.engine}, {ocr_seconds:.1f}s) ---", file=sys.stderr)
    for n, line in enumerate(ocr.lines, start=1):
        score = "" if line.confidence is None else f"  [{line.confidence:.2f}]"
        print(f"L{n}: {line.text}{score}", file=sys.stderr)

    start = time.perf_counter()
    result = extract(ocr, args.declared_type, args.country)
    print(f"--- extraction ({time.perf_counter() - start:.1f}s) ---\n", file=sys.stderr)

    output = result.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(output)

    flags = result.document.flags
    for flag in flags:
        print(f"FLAG {flag.code}: {flag.message}", file=sys.stderr)
    return 1 if any(f.code is FlagCode.EXTRACTION_FAILED for f in flags) else 0


if __name__ == "__main__":
    sys.exit(main())
