"""Run every synthetic case through OCR and extraction and compare with the answer key.

    uv run python scripts/run_synthetic.py [--gpu] [--only TEXT] [-o DIR]

Reads synthetic/ids/expected.json (written by synthetic/make_id_samples.py), runs
each case's image through OCR and `extract` with its declared type and country, and
checks every expected field value and the document's flags. One line per case goes
to stdout, with the wrong fields underneath; a total comes last.

- Values are compared by meaning, as the answer key asks: dates as dates, text
  without case, with ", " and " / " treated alike and spacing collapsed. Missing
  spaces inside a word ("ALEXJORDANEXAMPLE") still count as wrong.
- Flags are compared as a set of codes: every flag in "flags" must be there, and
  the only others allowed are those in "allowed_flags".
- PP-OCR on CPU is the default, as in the answer key's notes, and it leaves the GPU
  to Ollama. --gpu tries PaddleOCR-VL first, as ocr_image does.
- -o DIR writes each case's extraction JSON and numbered OCR text, plus
  summary.json, so a wrong field can be traced to the OCR or to the model.

Needs the OCR models (`uv sync --group ocr`), Ollama running, and OLLAMA_BASE_URL
and STRUCTURING_MODEL in .env or the environment. About 1 minute a case on CPU.

Exit code: 0 when every case matches, 1 when any doesn't, 2 when the answer key or
an image is missing.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ on the path, not the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.extract import extract
from api.ocr import OcrResult, ocr_image
from api.schemas import BaseDocument

ANSWER_KEY = ROOT / "synthetic" / "ids" / "expected.json"


@dataclass
class FieldResult:
    field: str
    ok: bool
    expected: Any
    got: Any
    confidence: str
    note: str | None


@dataclass
class CaseResult:
    case: str
    probes: str
    fields: list[FieldResult]
    flags_expected: list[str]
    flags_allowed: list[str]
    flags_got: list[str]
    flag_messages: list[str]
    seconds: float
    ok: bool = field(init=False)

    def __post_init__(self) -> None:
        self.ok = self.flags_ok and all(f.ok for f in self.fields)

    @property
    def flags_ok(self) -> bool:
        # Every expected flag must be there; allowed flags may be, nothing else.
        got, expected = set(self.flags_got), set(self.flags_expected)
        return expected <= got <= expected | set(self.flags_allowed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic documents through OCR and extraction and "
        "compare with synthetic/ids/expected.json."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="try PaddleOCR-VL on the GPU first (default: PP-OCR on CPU)",
    )
    parser.add_argument(
        "--only",
        metavar="TEXT",
        help="run only the cases whose name contains TEXT, e.g. 'permit'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="DIR",
        help="write each case's JSON and OCR text, and summary.json, here",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=ANSWER_KEY,
        metavar="FILE",
        help="answer key to score against (default synthetic/ids/expected.json; "
        "e.g. synthetic/caribbean/answers.json)",
    )
    return parser.parse_args(argv)


def case_name(index: int, case: dict[str, Any]) -> str:
    """Unique even when one image is run twice (the payslip declared as passport)."""
    return f"{index:02d}_{Path(case['image']).stem}_as_{case['declared_type']}"


def comparable(value: Any) -> str | None:
    """A value as compared with the answer key: "by meaning", not byte for byte."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).casefold().replace(" / ", ", ")
    return re.sub(r"\s*,\s*", ", ", " ".join(text.split()))


def same_value(got: Any, expected: Any) -> bool:
    return comparable(got) == comparable(expected)


def compare_fields(
    document: BaseDocument, expected: dict[str, Any]
) -> list[FieldResult]:
    results = []
    for name, want in expected.items():
        extracted = getattr(document, name)
        got = extracted.value
        results.append(
            FieldResult(
                field=name,
                ok=same_value(got, want),
                expected=want,
                got=got.isoformat() if isinstance(got, date) else got,
                confidence=str(extracted.confidence),
                note=extracted.note,
            )
        )
    return results


def numbered_ocr(ocr: OcrResult) -> str:
    return "\n".join(
        f"L{n}: {line.text}"
        + ("" if line.confidence is None else f"  [{line.confidence:.2f}]")
        for n, line in enumerate(ocr.lines, start=1)
    )


def report(result: CaseResult) -> str:
    good = sum(f.ok for f in result.fields)
    head = (
        f"{'PASS' if result.ok else 'FAIL'}  {result.case}: "
        f"{good}/{len(result.fields)} fields, "
        f"flags {'ok' if result.flags_ok else 'wrong'} ({result.seconds:.0f}s)"
    )
    lines = [head]
    for f in result.fields:
        if not f.ok:
            lines.append(
                f"      {f.field}: got {f.got!r} ({f.confidence}), "
                f"expected {f.expected!r}"
            )
            if f.note:
                lines.append(f"        note: {f.note}")
    if not result.flags_ok:
        lines.append(
            f"      flags: got {sorted(result.flags_got)}, "
            f"expected {sorted(result.flags_expected)}"
            + (
                f", allowed {sorted(result.flags_allowed)}"
                if result.flags_allowed
                else ""
            )
        )
        lines.extend(f"        {message}" for message in result.flag_messages)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.answer_key.is_file():
        print(
            f"error: no answer key at {args.answer_key}; run "
            "`uv run python synthetic/make_id_samples.py`",
            file=sys.stderr,
        )
        return 2
    cases = json.loads(args.answer_key.read_text(encoding="utf-8"))
    selected = [
        (case_name(i, case), case)
        for i, case in enumerate(cases)
        if args.only is None or args.only in case_name(i, case)
    ]
    if not selected:
        print(f"error: no case name contains {args.only!r}", file=sys.stderr)
        return 2
    missing = [c["image"] for _, c in selected if not (ROOT / c["image"]).is_file()]
    if missing:
        print(f"error: missing images: {', '.join(missing)}", file=sys.stderr)
        return 2
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for name, case in selected:
        start = time.perf_counter()
        ocr = ocr_image(ROOT / case["image"], use_gpu=args.gpu)
        extraction = extract(ocr, case["declared_type"], case["country"])
        document = extraction.document
        result = CaseResult(
            case=name,
            probes=case["probes"],
            fields=compare_fields(document, case["expected"]),
            flags_expected=list(case["flags"]),
            flags_allowed=list(case["allowed_flags"]),
            flags_got=[str(flag.code) for flag in document.flags],
            flag_messages=[f"{flag.code}: {flag.message}" for flag in document.flags],
            seconds=time.perf_counter() - start,
        )
        results.append(result)
        print(report(result), flush=True)

        if args.output:
            (args.output / f"{name}.json").write_text(
                extraction.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (args.output / f"{name}.ocr.txt").write_text(
                f"engine: {ocr.engine}\n{numbered_ocr(ocr)}\n", encoding="utf-8"
            )

    passed = sum(r.ok for r in results)
    print(f"\n{passed}/{len(results)} cases match the answer key")
    if args.output:
        (args.output / "summary.json").write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}", file=sys.stderr)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
