"""Keep synthetic/ids/expected.json consistent with the images and api/schemas.py."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from api import schemas
from api.schemas import DOCUMENT_MODELS, FlagCode, field_names

ROOT = Path(__file__).parent.parent
RUNS: list[dict[str, Any]] = json.loads(
    (ROOT / "synthetic" / "ids" / "expected.json").read_text(encoding="utf-8")
)

# Flags api.extract sets; validating a document can't produce them.
EXTRACT_FLAGS = {FlagCode.EXTRACTION_FAILED, FlagCode.DOCUMENT_TYPE_MISMATCH}


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 11, 22))


def run_id(run: dict[str, Any]) -> str:
    return f"{Path(run['image']).stem}-as-{run['declared_type']}"


@pytest.mark.parametrize("run", RUNS, ids=run_id)
def test_image_exists(run: dict[str, Any]) -> None:
    assert (ROOT / run["image"]).is_file()


@pytest.mark.parametrize("run", RUNS, ids=run_id)
def test_expected_values_validate_with_the_expected_flags(run: dict[str, Any]) -> None:
    model = DOCUMENT_MODELS[run["declared_type"]]
    assert set(run["expected"]) == set(field_names(model))

    doc = model.model_validate(
        {
            name: {"value": value, "confidence": "high" if value else "none"}
            for name, value in run["expected"].items()
        }
    )
    expected = {FlagCode(code) for code in run["flags"]} - EXTRACT_FLAGS
    assert {f.code for f in doc.flags} == expected
