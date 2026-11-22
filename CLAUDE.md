# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage KYC (know-your-customer) onboarding project. The only code so far is `api/schemas.py` and its tests. The folders `capture/`, `prompts/`, `reviewer/` and `synthetic/` are still empty placeholders. Don't assume an architecture beyond these directory names. Ask the user before deciding what goes where, and update this file as real modules appear.

## Extraction schemas (`api/schemas.py`)

These Pydantic v2 models validate extracted documents. Every extracted value is an `ExtractedField` with `value`, `raw_text`, `confidence` (high/medium/low/none) and `note`. Documents are a discriminated union on `document_type`. A `KYCPacket` holds several documents and runs cross-document checks.

The core rule is **errors vs flags**:
- **ValidationError** is for structurally wrong data: a value given with confidence "none", a date that can't be parsed day-first, a DOB giving an age outside 18–110, or unknown keys (all models use `extra="forbid"`).
- **Flags** are for things a reviewer should see. They never block loading. An expired document puts a `DOCUMENT_EXPIRED` flag on the document. Names scoring below 85 with rapidfuzz `token_sort_ratio`, or DOBs that aren't exactly equal, put `NAME_MISMATCH` or `DOB_MISMATCH` flags on the packet. Flags are rebuilt on every validation, so a dump→validate round trip doesn't duplicate them.
- Validators call the module-level `schemas.today()`. Tests monkeypatch it rather than depending on the real date.

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
- `uploads/` is ignored because it may hold real user identity documents. Never commit from it, and never copy its contents into fixtures.
- `.env`, `.env.*` and `secrets/` are git-ignored, and Claude is denied access to them in `.claude/settings.json`. Read configuration from environment variables instead of hard-coding it.
