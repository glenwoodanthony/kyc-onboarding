# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage KYC (know-your-customer) onboarding project. So far it has the OCR step in `api/ocr.py`, the extraction schemas in `api/schemas.py`, the extraction model's system prompt in `prompts/system_prompt.md` and the extraction call in `api/extract.py`, each with its own tests. `scripts/` holds developer tools that aren't part of the app.

## Synthetic documents (`synthetic/ids/`)

`uv run python synthetic/make_id_samples.py` renders 10 fictional documents for trying the pipeline (all seven types, plus damaged and edge-case variants) and writes `synthetic/ids/expected.json`. The same fictional person, ALEX JORDAN EXAMPLE, born 4 July 1990, is on every document.

- **Output is deterministic:** the noise is seeded, so regenerating gives byte-identical files.
- **`expected.json` is the answer key:** one entry per run, giving the image, the declared type and country to pass, what the run probes, the expected values (null where a field should come back empty) and the expected flags. Compare values by meaning. Addresses, for example, may be joined with ", " or " / ".
- **The cases probe specific things:** two-digit years, a two-line address, an expired passport with a machine-readable zone, an ambiguous `03/04/2026`, a prompt-injection line in the utility bill, a wrong-type run (payslip declared as passport), and degraded and creased permits.
- **What PP-OCR does on them:** it drops spaces and misreads text while still scoring 0.95 or more. Only local damage, like the crease, gives lines below 0.90.
- **`tests/test_synthetic_ids.py`** checks that every image exists and that the expected values validate against `api/schemas.py` with exactly the expected flags. It doesn't run OCR or the model. Change the generator and the key together.
- `synthetic/ocr_sample.png` (from `make_ocr_sample.py`) is separate. The `ocr` tests use it, so leave it alone. `synthetic/` holds the generated test image. The folders `capture/` and `reviewer/` are still empty placeholders. Don't assume an architecture beyond these directory names. Ask the user before deciding what goes where, and update this file as real modules appear.

## Extraction schemas (`api/schemas.py`)

These Pydantic v2 models validate extracted documents. Every extracted value is an `ExtractedField` with `value`, `raw_text`, `confidence` (high/medium/low/none) and `note`. Documents are a discriminated union on `document_type`. A `KYCPacket` holds several documents and runs cross-document checks.

The core rule is **errors vs flags**:
- **ValidationError** is for structurally wrong data: a value given with confidence "none", a date that can't be parsed day-first, a date value that its `raw_text` doesn't support (see below), a DOB giving an age outside 18–110, or unknown keys (all models use `extra="forbid"`).
- **Flags** are for things a reviewer should see. They never block loading. An expired document puts a `DOCUMENT_EXPIRED` flag on the document. Names scoring below 85 with rapidfuzz `token_sort_ratio`, or DOBs that aren't exactly equal, put `NAME_MISMATCH` or `DOB_MISMATCH` flags on the packet. Flags are rebuilt on every validation, so a dump→validate round trip doesn't duplicate them.
- **A date value must follow from its `raw_text`.** If `DateField.value` is set, `raw_text` must read, day-first, as exactly that date. A leading label (`Date of birth: …`) is allowed, as are ordinals and commas (`4th July, 1990`). This applies at every confidence level. It catches the model filling in what the document doesn't say. Both cases were seen live on qwen3:4b: a two-digit year `04.07.90` became `1990-07-04`, and a bare `2030` became `2030-01-01`. The error messages go back to the model on retry, so they tell it to set the value to null.
- **Every `raw_text` must be in the OCR text.** Inside `schemas.source_lines(lines)`, a document whose fields have a value with no `raw_text`, or a `raw_text` not found in `lines`, fails validation. Matching compares only letters and digits, ignoring case, so spacing repairs (`Name:ALEXJORDANEXAMPLE`) pass. Each ` / `-separated part may span one line or two consecutive lines, so a label on the line above its value still matches. Outside the block nothing is checked, so loading stored results still works. It catches a value invented together with a matching `raw_text`, which the date check can't. Live, qwen3:4b gave `issue_date` 2022-07-15 with `raw_text` `15.07.2022` for a card printed `15.01.2022`. The lines are held in a `ContextVar`, not passed through Instructor's `context=`, because that option also renders the messages as Jinja templates, and the OCR text comes from the customer.
- `api/extract.py` adds four more document flags: `EXTRACTION_FAILED`, `EXTRACTION_INCOMPLETE`, `DOCUMENT_TYPE_MISMATCH` and `PROMPT_INJECTION_SUSPECTED`. Only `DOCUMENT_EXPIRED` is rebuilt on validation; other document flags are kept as given.
- `KYCExtraction` is one extraction result: `declared_type`, `country` and `document`. `DocumentType`, `DOCUMENT_MODELS` (type → model) and `field_names(model)` are derived from the `Document` union.
- Validators call the module-level `schemas.today()`. Tests monkeypatch it rather than depending on the real date.

## Extraction prompt (`prompts/system_prompt.md`)

This is the system prompt for the extraction model. The model receives OCR text, not images. Its output is one JSON document that `api/schemas.py` validates.

- **Inputs:** `<declared_type>` and `<country>` say what the customer claimed. Country is only a hint for reading formats and must never fill a field. `<ocr_markdown>` holds the OCR text as Markdown, with each line prefixed `L1:`, `L2:` and so on. `<line_confidence>` is optional JSON, `[{"line": n, "confidence": 0–1}]`. `api.extract.format_user_message` builds all four.
- **Confidence caps:** a source line below 0.70 limits a field's confidence to "low", and one from 0.70 to below 0.90 limits it to "medium".
- **Multi-line addresses (rule 11):** include every line up to the town and postcode, joined with `, ` in `value` and ` / ` in `raw_text`. It was added because at temperature 0 qwen3:4b always kept only the first line of the address letter's address. New rules go at the end, so the existing rule numbers stay valid: the Output section refers to rules 7 and 8.
- **Wrong or unsupported documents:** the model always outputs the declared type. If the text isn't that type, every field is null/"none" with a note. There is no `unsupported_document` shape.
- **Kept in sync with the schema:** `tests/test_system_prompt.py` parses the prompt's field-list table and compares it with `api/schemas.py`. The ```json example must validate with no flags, and the prompt must never mention images. When you add or rename a document type or field, update the prompt's table and example along with the schema.

## Extraction (`api/extract.py`)

`extract(ocr, declared_type, country) -> KYCExtraction`. It sends the OCR text and the system prompt to a local model through Ollama's OpenAI-compatible endpoint, using Instructor (`Mode.JSON`).

- **Settings:** `Settings` (pydantic-settings) reads `OLLAMA_BASE_URL` (for example `http://localhost:11434/v1`) and `STRUCTURING_MODEL` (for example `qwen3:4b`) from the environment, then from `<repo>/.env`. There are no defaults in code, so a missing setting raises. Claude can't read or edit `.env`; the user maintains it.
- **`raw_text` is checked against the lines as sent:** `extract` wraps the model call and `_salvage` in `source_lines(_lines_as_sent(ocr))`. Those are the lines after withholding, so a value quoted from a withheld injection line is rejected too. Because the check runs inside Instructor's retry loop, the model gets the error on its next attempt. Tests that run `extract` use the `SOURCE` OCR fixture, which holds every `raw_text` the canned answers quote.
- **Pinned to the declared type:** Instructor validates against `DOCUMENT_MODELS[declared_type]` only, which keeps the schema small for a small model. Any `flags` the model outputs are dropped and the document is re-validated, so only code sets flags.
- **Type mismatch:** if none of the type's `IDENTIFYING_FIELDS` has a value, the document gets `DOCUMENT_TYPE_MISMATCH`. Those fields are its number or reference plus its defining date: for example `passport_number` and `date_of_birth`, or `employer_name` and `pay_date`. The old rule, "every field is none", missed a live case: a payslip declared as passport came back with only `full_name` filled. A real document with one of those fields unreadable isn't flagged. The weak spot is the address types: a utility bill declared as an address letter can fill `issuer` and `letter_date`, so it isn't flagged.
- **Prompt-injection defence:** the customer writes the document, so its text may address the model. The prompt says to ignore such text, but qwen3:4b obeyed "set … the provider to Verified Ltd" on the synthetic utility bill anyway.
  - `instruction_lines()` finds lines that match `INSTRUCTION_PATTERNS`. Those cover "ignore your rules", "note to the AI" (OCR often reads AI as `Al` or `A1`), a `SYSTEM:` or `AI:` prefix, "you are an AI", "mark … as verified", and any `<ocr_markdown>`-style tag. A match also takes in the lines that continue its sentence, up to `MAX_CONTINUATION_LINES`.
  - `format_user_message` replaces those lines with `WITHHELD`, keeping the line numbers. The document gets `PROMPT_INJECTION_SUSPECTED`, quoting the withheld lines, even when extraction fails.
  - Keep the patterns aimed at text written for a machine. "Please ignore previous reminders" and `Model:` labels must not match, and tests pin both.
  - Pattern matching can be beaten by rewording, so this raises the bar; it isn't a guarantee. The reviewer flag is the backstop.
- **Thinking is off:** requests send `reasoning_effort="none"`. With thinking on, qwen3:4b took many minutes per attempt on this machine. Without it a passport takes about 30 s. The prompt plus schema is about 3,300 tokens, against Ollama's default 4,096-token context. If output gets cut off, Instructor raises `IncompleteOutputException` and the result is the `EXTRACTION_FAILED` fallback. Raise the context with `OLLAMA_CONTEXT_LENGTH` on the Ollama server.
- **Temperature 0:** requests send `temperature=0`, so the same OCR text gets the same answer. With Ollama's default sampling, the same synthetic image gave different answers on different runs.
- **Retries:** `MAX_RETRIES = 2` is passed straight to Instructor. In Instructor 1.x `max_retries` counts retries, so the model is called at most 3 times, each retry carrying the previous validation errors.
- **Never raises on model failure, and keeps what was valid:** if the last attempt is still invalid, `_salvage` takes that answer (`InstructorRetryException.last_completion`) and keeps it field by field.
  - It starts from an all-null document and adds each field only if the document still validates. That way document-level checks, such as the DOB age range, reject the field that breaks them.
  - Rejected fields become null with confidence "none". The note reads `rejected by validation: <reason>`, and the model's `raw_text` is kept.
  - The document gets `EXTRACTION_INCOMPLETE`, which lists the rejected fields and why. The type-mismatch and injection checks still run on what's kept.
  - This replaced "throw everything away": live, qwen3:4b lost a correct name, number and expiry on the two-digit-year ID because one date stayed invalid.
  - Only when nothing can be kept (Ollama down or model not pulled, cut-off or non-JSON output, every field rejected) does it return every field at "none" with `EXTRACTION_FAILED`. Its reason is one short line, not Instructor's dump of every attempt.
- **Line confidence** goes to the model only for PP-OCR, whose Markdown is exactly its lines. VL Markdown is numbered without scores, skipping blank lines.
- **Tests** need no Ollama. They replace `_openai_client` with a real `openai.OpenAI` on an `httpx2.MockTransport` (`openai` 3.x is built on `httpx2`, not `httpx`), so Instructor's real parsing and retry loop run.

## Running the pipeline on one image (`scripts/extract_image.py`)

`uv run python scripts/extract_image.py IMAGE DECLARED_TYPE COUNTRY [--cpu] [-o FILE]` runs OCR and then extraction on one image, all locally.

- The JSON goes to stdout, or to FILE with `-o`. The OCR text goes to stderr, numbered `L1:`, `L2:` … with PP-OCR's confidence scores, so a wrong field can be traced to the OCR or to the model.
- `--cpu` passes `ocr_image(..., use_gpu=False)` and goes straight to PP-OCR. On this machine a run takes about 75 s that way, against about 150 s with VL, because VL and Ollama compete for the 4 GB GPU.
- Exit codes: 0 when a document was extracted (including `EXTRACTION_INCOMPLETE`), 1 for `EXTRACTION_FAILED`, 2 when the image is missing.
- The script puts the repo root on `sys.path` itself, because running a file puts only `scripts/` there.

## Running the synthetic cases (`scripts/run_synthetic.py`)

`uv run python scripts/run_synthetic.py [--gpu] [--only TEXT] [-o DIR]` runs every case in `synthetic/ids/expected.json` through OCR and extraction, then compares each expected value and the flags.

- **Output:** one PASS/FAIL line per case, with any wrong fields (value, confidence and note) underneath, then a total.
- **Comparison is by meaning:** dates as dates; text ignoring case, with `, ` and ` / ` treated alike and spacing collapsed. A missing space inside a word (`ALEXJORDANEXAMPLE`) still fails. Flags must match exactly, as a set of codes.
- **Engine:** PP-OCR on CPU by default; `--gpu` tries VL first.
- **Filtering:** `--only` keeps the cases whose name contains TEXT. Names are `NN_<image>_as_<declared type>`, because the payslip image is run twice.
- **Saved output:** `-o DIR` writes each case's extraction JSON and numbered OCR text, plus `summary.json`.
- **Speed and exit codes:** about 1 minute a case, so all 11 take about 10 minutes. Exit 0 when every case matches, 1 when any doesn't, 2 when the answer key or an image is missing.
- **Tests:** `tests/test_run_synthetic_script.py` stubs OCR and extraction.

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
