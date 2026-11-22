# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early scaffold for a KYC (know-your-customer) onboarding project. It has no application code yet. The top-level packages `api/`, `capture/`, `prompts/`, `reviewer/`, `synthetic/` and `tests/` each hold only a `.gitkeep`, and the README is empty. `pyproject.toml` declares no runtime dependencies, only dev tooling. Don't assume an architecture beyond these directory names. Ask the user before deciding what goes where, and update this file once real modules exist.

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
