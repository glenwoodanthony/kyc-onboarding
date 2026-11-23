"""Tests for api/ocr.py.

Unit tests stub out the engines and need no Paddle install. Integration tests (marker
`ocr`) run real OCR on synthetic/ocr_sample.png and are skipped without the ocr group.
"""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import ValidationError

from api import ocr
from api.ocr import Line, OcrResult, ocr_image

SAMPLE = Path(__file__).parent.parent / "synthetic" / "ocr_sample.png"


def fake_result(engine: Literal["paddleocr-vl", "pp-ocr"]) -> OcrResult:
    return OcrResult(markdown=f"from {engine}", lines=[], engine=engine)


@pytest.fixture(autouse=True)
def reset_vl_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with VL not yet known to be broken."""
    monkeypatch.setattr(ocr, "_vl_failure", None)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub both engines; record which ran. GPU defaults to unavailable."""
    called: list[str] = []

    def run_vl(path: Path) -> OcrResult:
        called.append("vl")
        return fake_result("paddleocr-vl")

    def run_ppocr(path: Path) -> OcrResult:
        called.append("ppocr")
        return fake_result("pp-ocr")

    monkeypatch.setattr(ocr, "_run_vl", run_vl)
    monkeypatch.setattr(ocr, "_run_ppocr", run_ppocr)
    monkeypatch.setattr(ocr, "_gpu_available", lambda: False)
    return called


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "doc.png"
    path.write_bytes(b"not really a png; the engines are stubbed")
    return path


# --- engine selection ------------------------------------------------------


def test_no_gpu_uses_ppocr_only(calls: list[str], image: Path) -> None:
    assert ocr_image(image).engine == "pp-ocr"
    assert calls == ["ppocr"]


def test_gpu_uses_vl(
    calls: list[str], image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ocr, "_gpu_available", lambda: True)
    assert ocr_image(image).engine == "paddleocr-vl"
    assert calls == ["vl"]


def test_use_gpu_false_skips_vl(
    calls: list[str], image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ocr, "_gpu_available", lambda: True)
    assert ocr_image(image, use_gpu=False).engine == "pp-ocr"
    assert calls == ["ppocr"]


@pytest.mark.parametrize(
    "error", [MemoryError("out of memory"), RuntimeError("CUDA error")]
)
def test_vl_failure_falls_back_to_ppocr(
    error: Exception,
    image: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing_vl(path: Path) -> OcrResult:
        raise error

    monkeypatch.setattr(ocr, "_gpu_available", lambda: True)
    monkeypatch.setattr(ocr, "_run_vl", failing_vl)
    monkeypatch.setattr(ocr, "_run_ppocr", lambda path: fake_result("pp-ocr"))

    with caplog.at_level(logging.WARNING, logger="api.ocr"):
        assert ocr_image(image).engine == "pp-ocr"
    assert str(error) in caplog.text


def test_vl_failure_is_remembered(
    image: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    vl_attempts: list[Path] = []

    def failing_vl(path: Path) -> OcrResult:
        vl_attempts.append(path)
        raise RuntimeError("cudnn64_9.dll not found")

    monkeypatch.setattr(ocr, "_gpu_available", lambda: True)
    monkeypatch.setattr(ocr, "_run_vl", failing_vl)
    monkeypatch.setattr(ocr, "_run_ppocr", lambda path: fake_result("pp-ocr"))

    with caplog.at_level(logging.WARNING, logger="api.ocr"):
        for _ in range(3):
            assert ocr_image(image).engine == "pp-ocr"
    assert len(vl_attempts) == 1
    assert caplog.text.count("cudnn64_9.dll") == 1


def test_missing_file_raises_before_any_engine(
    calls: list[str], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        ocr_image(tmp_path / "missing.png")
    assert calls == []


def test_accepts_str_path(calls: list[str], image: Path) -> None:
    assert ocr_image(str(image)).engine == "pp-ocr"


# --- models ----------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0, None])
def test_line_confidence_in_range_or_none(confidence: float | None) -> None:
    assert Line(text="x", confidence=confidence, bbox=None).confidence == confidence


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_line_confidence_out_of_range_is_an_error(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Line(text="x", confidence=confidence, bbox=None)


# --- converters (fake engine output) ---------------------------------------


def test_lines_from_ppocr_maps_text_score_and_box_in_order() -> None:
    res = {
        "rec_texts": ["PASSPORT", "Surname: EXAMPLE"],
        "rec_scores": [0.99, 0.87],
        "rec_boxes": [[10, 20, 200, 60], [10, 80, 400, 120]],
    }
    assert ocr._lines_from_ppocr(res) == [
        Line(text="PASSPORT", confidence=0.99, bbox=(10, 20, 200, 60)),
        Line(text="Surname: EXAMPLE", confidence=0.87, bbox=(10, 80, 400, 120)),
    ]


def test_lines_from_vl_blocks_split_lines_and_skip_images() -> None:
    # Mirrors paddlex PaddleOCRVLBlock: objects with .label, .content and .bbox.
    blocks = [
        SimpleNamespace(label="doc_title", content="PASSPORT", bbox=[10, 20, 200, 60]),
        SimpleNamespace(label="image", content="", bbox=[300, 20, 500, 200]),
        SimpleNamespace(
            label="text",
            content="Surname: EXAMPLE\nGiven names: ALEX JORDAN\n",
            bbox=[10, 80, 400, 160],
        ),
    ]
    assert ocr._lines_from_vl_blocks(blocks) == [
        Line(text="PASSPORT", confidence=None, bbox=(10, 20, 200, 60)),
        Line(text="Surname: EXAMPLE", confidence=None, bbox=(10, 80, 400, 160)),
        Line(text="Given names: ALEX JORDAN", confidence=None, bbox=(10, 80, 400, 160)),
    ]


# --- NVIDIA DLL folders (Windows cuDNN from PyPI) -------------------------


def test_nvidia_dll_dirs_finds_each_library_bin(tmp_path: Path) -> None:
    for lib in ("cudnn", "cublas"):
        (tmp_path / lib / "bin").mkdir(parents=True)
    (tmp_path / "no_bin_here").mkdir()
    assert ocr._nvidia_dll_dirs([str(tmp_path)]) == [
        tmp_path / "cublas" / "bin",
        tmp_path / "cudnn" / "bin",
    ]


def test_nvidia_dll_dirs_empty_without_roots() -> None:
    assert ocr._nvidia_dll_dirs([]) == []


@pytest.mark.skipif(sys.platform != "win32", reason="DLL search path is Windows-only")
def test_expose_dll_dirs_prepends_to_path_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", r"C:\existing")
    monkeypatch.setattr(ocr, "_dll_handles", [])
    dirs = [tmp_path / "a", tmp_path / "b"]
    for d in dirs:
        d.mkdir()
    ocr._expose_dll_dirs(dirs)
    ocr._expose_dll_dirs(dirs)
    entries = os.environ["PATH"].split(os.pathsep)
    assert entries == [str(dirs[1]), str(dirs[0]), r"C:\existing"]
    assert len(ocr._dll_handles) == 2


# --- integration: real OCR on the synthetic image --------------------------


@pytest.mark.ocr
def test_vl_runs_on_gpu_when_available() -> None:
    """Fails if VL silently falls back, e.g. because cuDNN can't be found."""
    pytest.importorskip("paddleocr")
    if not ocr._gpu_available():
        pytest.skip("no CUDA GPU")
    result = ocr_image(SAMPLE)
    assert ocr._vl_failure is None
    assert result.engine == "paddleocr-vl"
    text = result.markdown.upper()
    for expected in ("EXAMPLE", "FICTIONLAND", "04/07/1990"):
        assert expected in text
    assert all(line.confidence is None for line in result.lines)


@pytest.mark.ocr
def test_cpu_fallback_reads_synthetic_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("paddleocr")
    monkeypatch.setattr(ocr, "_gpu_available", lambda: False)
    result = ocr_image(SAMPLE)
    assert result.engine == "pp-ocr"
    text = result.markdown.upper()
    for expected in ("EXAMPLE", "FICTIONLAND", "04/07/1990"):
        assert expected in text
    assert result.lines
    assert all(
        line.confidence is not None and 0 < line.confidence <= 1
        for line in result.lines
    )


@pytest.mark.ocr
def test_default_path_reads_synthetic_image() -> None:
    pytest.importorskip("paddleocr")
    result = ocr_image(SAMPLE)
    print(f"engine used: {result.engine}")
    text = result.markdown.upper()
    assert "EXAMPLE" in text
    assert "04/07/1990" in text
