"""Keep prompts/system_prompt.md in sync with api/schemas.py."""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel, TypeAdapter

from api import schemas
from api.schemas import Document

PROMPT = (Path(__file__).parent.parent / "prompts" / "system_prompt.md").read_text(
    encoding="utf-8"
)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 6, 15))


def document_models() -> list[type[BaseModel]]:
    union = get_args(Document.__value__)[0]
    return list(get_args(union))


def schema_fields() -> dict[str, set[str]]:
    """document_type -> field names the model must output."""
    fields: dict[str, set[str]] = {}
    for model in document_models():
        doc_type = model.model_fields["document_type"].default
        fields[doc_type] = set(model.model_fields) - {"document_type", "flags"}
    return fields


def prompt_field_table() -> dict[str, set[str]]:
    """Parse the '| `type` | `field`, ... |' rows of the prompt's field list."""
    table: dict[str, set[str]] = {}
    for line in PROMPT.splitlines():
        match = re.match(r"\| `(\w+)` \|(.*)\|$", line)
        if match:
            table[match.group(1)] = set(re.findall(r"`(\w+)`", match.group(2)))
    return table


def example_output() -> Any:
    blocks = re.findall(r"```json\n(.*?)\n```", PROMPT, re.DOTALL)
    assert len(blocks) == 1, "expected exactly one ```json example block"
    return json.loads(blocks[0])


def test_field_list_matches_schema() -> None:
    assert prompt_field_table() == schema_fields()


def test_example_output_validates_without_flags() -> None:
    doc = TypeAdapter(Document).validate_python(example_output())
    assert doc.flags == []


def test_example_reads_dates_day_first() -> None:
    assert example_output()["date_of_birth"]["value"] == "1990-07-04"


def test_prompt_does_not_mention_images() -> None:
    assert re.search(r"image", PROMPT, re.IGNORECASE) is None


def test_prompt_describes_both_inputs() -> None:
    assert "<ocr_markdown>" in PROMPT
    assert "<line_confidence>" in PROMPT
