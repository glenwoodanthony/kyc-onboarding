# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage KYC (know-your-customer) onboarding project. So far it has the OCR step in `api/ocr.py`, the extraction schemas in `api/schemas.py`, the extraction model's system prompt in `prompts/system_prompt.md` and the extraction call in `api/extract.py`, each with its own tests. `scripts/` holds developer tools that aren't part of the app. `synthetic/` holds the generated test image. The folders `capture/` and `reviewer/` are still empty placeholders. Don't assume an architecture beyond these directory names. Ask the user before deciding what goes where, and update this file as real modules appear.

## Extraction schemas (`api/schemas.py`)

These Pydantic v2 models validate extracted documents. Every extracted value is an `ExtractedField` with `value`, `raw_text`, `confidence` (high/medium/low/none) and `note`. Documents are a discriminated union on `document_type`. A `KYCPacket` holds several documents and runs cross-document checks.

The core rule is **errors vs flags**:
- **ValidationError** is for structurally wrong data: a value given with confidence "none", a date that can't be parsed day-first, a DOB giving an age outside 18–110, or unknown keys (all models use `extra="forbid"`).
- **Flags** are for things a reviewer should see. They never block loading. An expired document puts a `DOCUMENT_EXPIRED` flag on the document. Names scoring below 85 with rapidfuzz `token_sort_ratio`, or DOBs that aren't exactly equal, put `NAME_MISMATCH` or `DOB_MISMATCH` flags on the packet. Flags are rebuilt on every validation, so a dump→validate round trip doesn't duplicate them.
- `api/extract.py` adds two more document flags: `EXTRACTION_FAILED` and `DOCUMENT_TYPE_MISMATCH`. Only `DOCUMENT_EXPIRED` is rebuilt on validation; other document flags are kept as given.
- `KYCExtraction` is one extraction result: `declared_type`, `country` and `document`. `DocumentType`, `DOCUMENT_MODELS` (type → model) and `field_names(model)` are derived from the `Document` union.
- Validators call the module-level `schemas.today()`. Tests monkeypatch it rather than depending on the real date.

## Extraction prompt (`prompts/system_prompt.md`)

This is the system prompt for the extraction model. The model receives OCR text, not images. Its output is one JSON document that `api/schemas.py` validates.

- **Inputs:** `<declared_type>` and `<country>` say what the customer claimed. Country is only a hint for reading formats and must never fill a field. `<ocr_markdown>` holds the OCR text as Markdown, with each line prefixed `L1:`, `L2:` and so on. `<line_confidence>` is optional JSON, `[{"line": n, "confidence": 0–1}]`. `api.extract.format_user_message` builds all four.
- **Confidence caps:** a source line below 0.70 limits a field's confidence to "low", and one from 0.70 to below 0.90 limits it to "medium".
- **Wrong or unsupported documents:** the model always outputs the declared type. If the text isn't that type, every field is null/"none" with a note. There is no `unsupported_document` shape.
- **Kept in sync with the schema:** `tests/test_system_prompt.py` parses the prompt's field-list table and compares it with `api/schemas.py`. The ```json example must validate with no flags, and the prompt must never mention images. When you add or rename a document type or field, update the prompt's table and example along with the schema.

## Extraction (`api/extract.py`)

`extract(ocr, declared_type, country) -> KYCExtraction`. It sends the OCR text and the system prompt to a local model through Ollama's OpenAI-compatible endpoint, using Instructor (`Mode.JSON`).

- **Settings:** `Settings` (pydantic-settings) reads `OLLAMA_BASE_URL` (for example `http://localhost:11434/v1`) and `STRUCTURING_MODEL` (for example `qwen3:4b`) from the environment, then from `<repo>/.env`. There are no defaults in code, so a missing setting raises. Claude can't read or edit `.env`; the user maintains it.
- **Pinned to the declared type:** Instructor validates against `DOCUMENT_MODELS[declared_type]` only, which keeps the schema small for a small model. Any `flags` the model outputs are dropped and the document is re-validated, so only code sets flags. If every field comes back "none", the document gets `DOCUMENT_TYPE_MISMATCH`.
- **Thinking is off:** requests send `reasoning_effort="none"`. With thinking on, qwen3:4b took many minutes per attempt on this machine. Without it a passport takes about 30 s. The prompt plus schema is about 3,300 tokens, against Ollama's default 4,096-token context. If output gets cut off, Instructor raises `IncompleteOutputException` and the result is the `EXTRACTION_FAILED` fallback. Raise the context with `OLLAMA_CONTEXT_LENGTH` on the Ollama server.
- **Retries:** `MAX_RETRIES = 2` is passed straight to Instructor. In Instructor 1.x `max_retries` counts retries, so the model is called at most 3 times, each retry carrying the previous validation errors.
- **Never raises on model failure:** on `InstructorRetryException`, unparseable output or any `openai.OpenAIError` (Ollama down, model not pulled), it returns the declared type with every field at confidence "none" and one `EXTRACTION_FAILED` flag giving the reason.
- **Line confidence** goes to the model only for PP-OCR, whose Markdown is exactly its lines. VL Markdown is numbered without scores, skipping blank lines.
- **Tests** need no Ollama. They replace `_openai_client` with a real `openai.OpenAI` on an `httpx2.MockTransport` (`openai` 3.x is built on `httpx2`, not `httpx`), so Instructor's real parsing and retry loop run.

## Running the pipeline on one image (`scripts/extract_image.py`)

`uv run python scripts/extract_image.py IMAGE DECLARED_TYPE COUNTRY [--cpu] [-o FILE]` runs OCR and then extraction on one image, all locally.

- The JSON goes to stdout, or to FILE with `-o`. The OCR text goes to stderr, numbered `L1:`, `L2:` … with PP-OCR's confidence scores, so a wrong field can be traced to the OCR or to the model.
- `--cpu` passes `ocr_image(..., use_gpu=False)` and goes straight to PP-OCR. On this machine a run takes about 75 s that way, against about 150 s with VL, because VL and Ollama compete for the 4 GB GPU.
- Exit codes: 0 when a document was extracted, 1 for `EXTRACTION_FAILED`, 2 when the image is missing.
- The script puts the repo root on `sys.path` itself, because running a file puts only `scripts/` there.

## OCR (`api/ocr.py`)

`ocr_image(path, *, use_gpu=True) -> OcrResult(markdown, lines, engine)`. Each `Line` has `text`, `confidence` (0–1 or None) and `bbox` (x0, y0, x1, y1). `use_gpu=False` skips VL and goes straight to PP-OCR.

- **Engines:** with a usable GPU it runs PaddleOCR-VL first. If there's no GPU, or VL raises for any reason, it uses classic PP-OCR (PP-OCRv5 mobile models) on CPU. A VL failure is remembered in `_vl_failure` for the rest of the process. VL lines have `confidence=None` because VL gives no per-line score; PP-OCR lines have recognition scores.
- **Optional dependency:** Paddle lives in the `ocr` dependency group. Install it with `uv sync --group ocr`. A plain `uv sync` uninstalls it again. `paddlepaddle-gpu` comes from PaddlePaddle's own CUDA 12.6 index, declared in `pyproject.toml`, and has no hashes in `uv.lock`. Paddle imports are lazy, so `api.ocr` imports without the group.
- **Tests:** unit tests stub the engines. Tests marked `ocr` run real models on `synthetic/ocr_sample.png` and are skipped without the group. Skip them with `-m "not ocr"`, because they take about 2 minutes with VL on the GPU. Regenerate the image with `uv run python synthetic/make_ocr_sample.py`.
- **cuDNN on Windows:** the Paddle GPU wheel doesn't bundle cuDNN. The `ocr` group installs `nvidia-cudnn-cu12==9.5.1.17` (plus cuBLAS) from PyPI, Windows-only, and `_expose_dll_dirs` puts `site-packages/nvidia/*/bin` on `PATH` before Paddle loads. Without it, VL fails on a missing `cudnn64_9.dll` and quietly falls back. The `ocr` test `test_vl_runs_on_gpu_when_available` catches that.
- **This machine (RTX 2050, 4 GB):** VL runs on the GPU but takes about 40 s per page, against about 1 s for PP-OCR, because Paddle uses about 6 GB and Windows spills the excess into system RAM. Models are cached in `~/.paddlex/official_models`. The first download needed the Windows certificate store (`truststore`), because Python's HTTPS doesn't trust the certificate that intercepts traffic on this machine.

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
