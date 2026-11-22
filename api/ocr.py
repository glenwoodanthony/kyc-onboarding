"""OCR for KYC documents: PaddleOCR-VL on GPU, classic PP-OCR on CPU as the fallback.

`ocr_image(path)` returns the page as Markdown plus its text lines, the input the
extraction prompt (prompts/system_prompt.md) expects.

- With a usable GPU, PaddleOCR-VL runs first. It gives better Markdown (layout,
  reading order, tables) but no per-line confidence, so its lines have
  `confidence=None`.
- Without a GPU, or if VL fails for any reason (out of memory, missing CUDA/cuDNN
  libraries), PP-OCR runs on CPU. Its lines carry recognition scores from 0 to 1.
  A VL failure is remembered for the rest of the process so it isn't retried on
  every call.

Paddle is an optional dependency (`uv sync --group ocr`) and is imported lazily, so
this module imports without it.
"""

import logging
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# PP-OCRv5 mobile models: small (about 21 MB) and fast enough on CPU.
PPOCR_DET_MODEL = "PP-OCRv5_mobile_det"
PPOCR_REC_MODEL = "PP-OCRv5_mobile_rec"

# VL layout labels that hold pictures, not text.
VL_NON_TEXT_LABELS = frozenset({"image", "header_image", "footer_image", "chart"})

Engine = Literal["paddleocr-vl", "pp-ocr"]


class Line(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    # Recognition score from 0 to 1; None when the engine gives none (PaddleOCR-VL).
    confidence: float | None = Field(ge=0, le=1)
    # Axis-aligned box in image pixels: x0, y0, x1, y1.
    bbox: tuple[float, float, float, float] | None


class OcrResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: str
    lines: list[Line]
    engine: Engine


# Why VL failed, once it has; VL isn't retried after that.
_vl_failure: str | None = None


def ocr_image(path: str | Path) -> OcrResult:
    """OCR one image. Tries PaddleOCR-VL on GPU, falls back to PP-OCR on CPU."""
    global _vl_failure
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if _vl_failure is None and _gpu_available():
        try:
            return _run_vl(path)
        except Exception as exc:  # noqa: BLE001 - any VL failure means: use the fallback
            _vl_failure = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "PaddleOCR-VL failed, using PP-OCR on CPU from now on: %s", _vl_failure
            )
    return _run_ppocr(path)


def _gpu_available() -> bool:
    try:
        import paddle  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return (
        bool(paddle.device.is_compiled_with_cuda())
        and paddle.device.cuda.device_count() > 0
    )


# --- engines ---------------------------------------------------------------


@cache
def _vl_pipeline() -> Any:
    from paddleocr import PaddleOCRVL  # pyright: ignore[reportMissingImports]

    return PaddleOCRVL(device="gpu:0")


@cache
def _ppocr_pipeline() -> Any:
    from paddleocr import PaddleOCR  # pyright: ignore[reportMissingImports]

    return PaddleOCR(
        device="cpu",
        text_detection_model_name=PPOCR_DET_MODEL,
        text_recognition_model_name=PPOCR_REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _run_vl(path: Path) -> OcrResult:
    res = _vl_pipeline().predict(str(path))[0]
    return OcrResult(
        markdown=res.markdown["markdown_texts"],
        lines=_lines_from_vl_blocks(res["parsing_res_list"]),
        engine="paddleocr-vl",
    )


def _run_ppocr(path: Path) -> OcrResult:
    res = _ppocr_pipeline().predict(str(path))[0]
    lines = _lines_from_ppocr(res)
    return OcrResult(
        markdown="\n".join(line.text for line in lines), lines=lines, engine="pp-ocr"
    )


# --- converters (pure, unit-tested with fake engine output) ----------------


def _bbox(box: Any) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (
        float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)
    )
    return x0, y0, x1, y1


def _lines_from_ppocr(res: Mapping[str, Any]) -> list[Line]:
    """PP-OCR result -> one Line per recognised text line, in detection order."""
    return [
        Line(text=text, confidence=float(score), bbox=_bbox(box))
        for text, score, box in zip(
            res["rec_texts"], res["rec_scores"], res["rec_boxes"], strict=True
        )
    ]


def _lines_from_vl_blocks(blocks: Iterable[Any]) -> list[Line]:
    """VL layout blocks (.label, .content, .bbox) -> one Line per non-empty text line."""
    lines: list[Line] = []
    for block in blocks:
        if block.label in VL_NON_TEXT_LABELS:
            continue
        bbox = _bbox(block.bbox)
        lines.extend(
            Line(text=text.strip(), confidence=None, bbox=bbox)
            for text in str(block.content).splitlines()
            if text.strip()
        )
    return lines
