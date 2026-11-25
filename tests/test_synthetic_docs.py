"""The synthetic-docs skill's generator: answer key, determinism, watermark, packets."""

import importlib.util
import json
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image
from pydantic import TypeAdapter

from api import schemas
from api.schemas import DOCUMENT_MODELS, FlagCode, KYCPacket, field_names

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / ".claude" / "skills" / "synthetic-docs" / "scripts" / "generate.py"
COMMITTED = {
    False: ROOT / "synthetic" / "caribbean",
    True: ROOT / "synthetic" / "caribbean_mismatch",
}
RUNNER_KEYS = {
    "image",
    "declared_type",
    "country",
    "probes",
    "expected",
    "flags",
    "allowed_flags",
}


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("synthetic_docs_generate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_generator()


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas, "today", lambda: date(2026, 11, 22))


@pytest.fixture(scope="module", params=[False, True], ids=["default", "mismatch"])
def generated(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    out = tmp_path_factory.mktemp("mismatch" if request.param else "default")
    gen.write(gen.build(mismatch=request.param), out)
    return out


def answers(out: Path) -> list[dict[str, Any]]:
    return json.loads((out / "answers.json").read_text(encoding="utf-8"))


DOCUMENT: TypeAdapter[schemas.Document] = TypeAdapter(schemas.Document)


def document(entry: dict[str, Any]) -> schemas.Document:
    model = DOCUMENT_MODELS[entry["declared_type"]]
    assert set(entry["expected"]) == set(field_names(model))
    fields = {
        name: {"value": value, "raw_text": value, "confidence": "high"}
        for name, value in entry["expected"].items()
    }
    return DOCUMENT.validate_python({"document_type": entry["declared_type"], **fields})


def test_writes_seven_images_and_a_runner_compatible_key(generated: Path) -> None:
    entries = answers(generated)
    assert [Path(e["image"]).name for e in entries] == [
        "national_id.png",
        "birth_certificate_typed.png",
        "birth_certificate_handwritten.png",
        "utility_bill.png",
        "utility_bill_relative.png",
        "jp_letter.png",
        "payslip.png",
    ]
    for entry in entries:
        assert RUNNER_KEYS <= set(entry)
        assert Path(entry["image"]).is_file()


def test_expected_values_validate_without_flags(generated: Path) -> None:
    for entry in answers(generated):
        assert document(entry).flags == []
        assert entry["flags"] == entry["allowed_flags"] == []


def test_name_mismatch_follows_matches_applicant(generated: Path) -> None:
    entries = answers(generated)
    national_id = document(entries[0])
    assert entries[0]["declared_type"] == "national_id"
    for entry in entries[1:]:
        packet = KYCPacket(documents=[national_id, document(entry)])
        flagged = FlagCode.NAME_MISMATCH in {f.code for f in packet.flags}
        assert flagged == (not entry["matches_applicant"]), entry["image"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_mismatch_changes_only_the_birth_certificate_names(mismatch: bool) -> None:
    entries = {Path(e["image"]).name: e for e in answers(COMMITTED[mismatch])}
    certs = [e for e in entries.values() if e["declared_type"] == "birth_certificate"]
    assert all(e["matches_applicant"] is not mismatch for e in certs)
    assert entries["utility_bill_relative.png"]["matches_applicant"] is False

    other = {Path(e["image"]).name: e for e in answers(COMMITTED[not mismatch])}
    for name, entry in entries.items():
        diff = {
            k
            for k in entry["expected"]
            if entry["expected"][k] != other[name]["expected"][k]
        }
        assert diff == (
            {"full_name"} if name.startswith("birth_certificate") else set()
        )


def test_output_is_deterministic(tmp_path: Path) -> None:
    gen.write(gen.build(seed=3), tmp_path / "a")
    gen.write(gen.build(seed=3), tmp_path / "b")
    for path in (tmp_path / "a").iterdir():
        if path.suffix == ".png":
            assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes()


@pytest.mark.parametrize("mismatch", [False, True])
def test_committed_set_matches_the_generator(mismatch: bool, tmp_path: Path) -> None:
    # Regenerate with: uv run python .claude/skills/synthetic-docs/scripts/generate.py [--mismatch]
    gen.write(gen.build(mismatch=mismatch), tmp_path)
    committed = COMMITTED[mismatch]
    fresh, kept = answers(tmp_path), answers(committed)
    for entry in fresh + kept:
        entry["image"] = Path(entry["image"]).name
    assert fresh == kept
    for entry in kept:
        name = entry["image"]
        assert (tmp_path / name).read_bytes() == (committed / name).read_bytes(), name


def test_every_saved_image_carries_the_watermark(tmp_path: Path) -> None:
    docs = gen.build()
    gen.write(docs, tmp_path)
    for doc in docs:
        saved = Image.open(tmp_path / doc.image).convert("RGB")
        assert saved.tobytes() == gen.stamp_specimen(doc.picture).tobytes()
        assert saved.tobytes() != doc.picture.convert("RGB").tobytes()
