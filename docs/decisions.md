# Architectural Decisions

## AD-0001: Project layout and packaging
- Decision: Use `src/` layout with top-level modules [src/lib](src/lib) and [src/cli](src/cli) to match project rules in [.roo/rules/15-PythonGUI.md](.roo/rules/15-PythonGUI.md). Keep `distribution = false` in [pyproject.toml](pyproject.toml) for a simple app-style workflow during PoC.
- Rationale: Minimizes packaging complexity. PDM scripts set `PYTHONPATH=src` to allow `python -m cli.main` execution.
- Consequences: Import paths are simple (`import lib.config`); if packaging is desired later, we can refactor into a namespaced package without breaking CLI semantics.

## AD-0002: HTTP client choice
- Decision: Use `httpx` for HTTP calls to Hugging Face Serverless Inference.
- Rationale: Modern API, good typing, sync client suffices for PoC.
- Alternatives: `requests` — simpler but `httpx` offers better ergonomics for future async if needed.