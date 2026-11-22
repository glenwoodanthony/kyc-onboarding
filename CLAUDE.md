# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage KYC (know-your-customer) onboarding project. So far it has the OCR step in `api/ocr.py`, the extraction schemas in `api/schemas.py` and the extraction model's system prompt in `prompts/system_prompt.md`, each with its own tests. `synthetic/` holds the generated test image. The folders `capture/` and `reviewer/` are still empty placeholders. Don't assume an architecture beyond these directory names. Ask the user before deciding what goes where, and update this file as real modules appear.

## Extraction schemas (`api/schemas.py`)

These Pydantic v2 models validate extracted documents. Every extracted value is an `ExtractedField` with `value`, `raw_text`, `confidence` (high/medium/low/none) and `note`. Documents are a discriminated union on `document_type`. A `KYCPacket` holds several documents and runs cross-document checks.

The core rule is **errors vs flags**:
- **ValidationError** is for structurally wrong data: a value given with confidence "none", a date that can't be parsed day-first, a DOB giving an age outside 18–110, or unknown keys (all models use `extra="forbid"`).
- **Flags** are for things a reviewer should see. They never block loading. An expired document puts a `DOCUMENT_EXPIRED` flag on the document. Names scoring below 85 with rapidfuzz `token_sort_ratio`, or DOBs that aren't exactly equal, put `NAME_MISMATCH` or `DOB_MISMATCH` flags on the packet. Flags are rebuilt on every validation, so a dump→validate round trip doesn't duplicate them.
- Validators call the module-level `schemas.today()`. Tests monkeypatch it rather than depending on the real date.

## Extraction prompt (`prompts/system_prompt.md`)

This is the system prompt for the extraction model. The model receives OCR text, not images. Its output is one JSON document that `api/schemas.py` validates.

- **Inputs:** `<ocr_markdown>` holds the OCR text as Markdown, with each line prefixed `L1:`, `L2:` and so on. `<line_confidence>` is optional JSON, `[{"line": n, "confidence": 0–1}]`. The capture step, which doesn't exist yet, must produce exactly this format.
- **Confidence caps:** a source line below 0.70 limits a field's confidence to "low", and one from 0.70 to below 0.90 limits it to "medium".
- **Unsupported documents:** for anything outside the seven types, the model returns `{"unsupported_document": "<reason>"}`. That shape isn't in the schema, so callers must check for it before validating.
- **Kept in sync with the schema:** `tests/test_system_prompt.py` parses the prompt's field-list table and compares it with `api/schemas.py`. The ```json example must validate with no flags, and the prompt must never mention images. When you add or rename a document type or field, update the prompt's table and example along with the schema.

## OCR (`api/ocr.py`)

`ocr_image(path) -> OcrResult(markdown, lines, engine)`. Each `Line` has `text`, `confidence` (0–1 or None) and `bbox` (x0, y0, x1, y1).

- **Engines:** with a usable GPU it runs PaddleOCR-VL first. If there's no GPU, or VL raises for any reason, it uses classic PP-OCR (PP-OCRv5 mobile models) on CPU. A VL failure is remembered in `_vl_failure` for the rest of the process. VL lines have `confidence=None` because VL gives no per-line score; PP-OCR lines have recognition scores.
- **Optional dependency:** Paddle lives in the `ocr` dependency group. Install it with `uv sync --group ocr`. A plain `uv sync` uninstalls it again. `paddlepaddle-gpu` comes from PaddlePaddle's own CUDA 12.6 index, declared in `pyproject.toml`, and has no hashes in `uv.lock`. Paddle imports are lazy, so `api.ocr` imports without the group.
- **Tests:** unit tests stub the engines. Tests marked `ocr` run real models on `synthetic/ocr_sample.png` and are skipped without the group. Skip them with `-m "not ocr"`, because they take about 30 s. Regenerate the image with `uv run python synthetic/make_ocr_sample.py`.
- **This machine:** PaddleOCR-VL currently fails because cuDNN 9 (`cudnn64_9.dll`) isn't installed, so the PP-OCR fallback is what actually runs. Models are cached in `~/.paddlex/official_models`. The first download needed the Windows certificate store (`truststore`), because Python's HTTPS doesn't trust the certificate that intercepts traffic on this machine.

## Tooling

- Python 3.12 (`.python-version`), managed with **uv**. Run every tool through `uv run` so it uses the versions pinned in `uv.lock`.
- Dev dependencies: pytest, ruff, pyright, pre-commit. Add packages with `uv add <pkg>`, or `uv add --dev <pkg>` for dev tools.
- The shell is configured as PowerShell (`.claude/settings.json`).

## Commands

```
uv sync                              # install/refresh the environment
uv run pytest                        # run all tests
uv run pytest tests/test_x.py::test_name   # run a single test
uv run ruff check .                  # lint (add --fix to autofix)
uv run ruff format .                 # format
uv run pyright                       # type-check the whole project
uv run pre-commit run --all-files    # run every pre-commit hook
```

## Pre-commit hooks

The hooks in `.pre-commit-config.yaml` run in this order: **gitleaks** (secret scanning), ruff check, ruff format, then pyright. Pyright checks the whole project rather than only the staged files (`pass_filenames: false`). The ruff and pyright hooks are `language: system` and call `uv run`, so keep their versions in step through `uv.lock`, not through hook `rev`s. Never bypass the hooks with `--no-verify`.

## Handling identity data

This project deals with identity documents, so the repo is set up to keep real personal data out of git:

- `.gitignore` blocks `*.jpg`, `*.png` and `*.pdf` everywhere **except under `synthetic/`**. Test fixtures and sample documents must be synthetic and live in `synthetic/`.
- Test data and prompt examples use fictional data only: made-up people, and placeholder places and institutions such as Exampletown, Fictionland and Example Bank. Don't use real place, bank or company names.
- `uploads/` is ignored because it may hold real user identity documents. Never commit from it, and never copy its contents into fixtures.
- `.env`, `.env.*` and `secrets/` are git-ignored, and Claude is denied access to them in `.claude/settings.json`. Read configuration from environment variables instead of hard-coding it.
