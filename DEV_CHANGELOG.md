# Development Change Log (Record Every Change)

> Use this log to record **every change**, even small ones.
> Keep entries short but precise, and always include **what changed**, **why**, and **how it was tested**.

---

## Format (copy/paste for each entry)

```md
### YYYY-MM-DD HH:MM (local time) — <short title>

**Scope:** backend | ml | infra | docs | repo
**Type:** feat | fix | refactor | chore | test | docs

**What changed**
- 

**Why**
- 

**Files touched**
- 

**How tested**
- [ ] Unit tests: 
- [ ] Integration tests: 
- [ ] Manual checks: 
- [ ] Performance checks: 

**Notes / Follow-ups**
- 
```

---

## Entries

### 2026-01-28 12:40 — Step B1: Backend skeleton (FastAPI) + tests

**Scope:** backend
**Type:** feat

**What changed**
- Created backend service skeleton under `services/api/`.
- Added FastAPI app with `GET /health` and `GET /version`.
- Added pytest suite for health/version contract and 404 behavior.
- Added `__init__.py` files so imports work consistently.

**Why**
- Establish a testable backend foundation before adding DB, uploads, or ML inference.

**Files touched**
- `services/api/requirements.txt`
- `services/api/app/main.py`
- `services/api/app/__init__.py`
- `services/api/tests/test_api.py`
- `services/__init__.py`
- `services/api/__init__.py`

**How tested**
- [x] Unit tests: `py -3 -m pytest -q services/api/tests`
  - Result: `3 passed`

**Notes / Follow-ups**
- Next step (B2): add DB models + repository layer.

---

### 2026-01-28 12:45 — Step B2 prep: Add SQLModel dependency

**Scope:** backend
**Type:** chore

**What changed**
- Added `sqlmodel` to backend dependencies.

**Why**
- Prepare for Step B2 (DB models + repository layer).

**Files touched**
- `services/api/requirements.txt`

**How tested**
- [ ] Unit tests: pending (dependency-only change)

**Notes / Follow-ups**
- Next: implement `AnalysisRecord` SQLModel + SQLite engine + repository tests.

---

### 2026-01-28 13:05 — Step B2: Initialize DB on API startup

**Scope:** backend
**Type:** feat

**What changed**
- Added DB initialization on FastAPI startup (`create_db_and_tables()`).
- Added a test to verify table creation in an isolated temporary SQLite DB.

**Why**
- Ensure the service is runnable without manual DB setup during early development.

**Files touched**
- `services/api/app/main.py`
- `services/api/tests/test_startup_db_init.py`

**How tested**
- [x] Unit tests: `py -3 -m pytest -q services/api/tests`
  - Result: `6 passed`

**Notes / Follow-ups**
- Migrated deprecated `@app.on_event("startup")` to FastAPI lifespan handler to remove deprecation warnings and future-proof startup initialization.

---

### 2026-01-28 12:55 — Step B2: SQLModel DB layer (AnalysisRecord + SQLite) + repository

**Scope:** backend
**Type:** feat

**What changed**
- Added `AnalysisRecord` SQLModel schema for `analysis_records`.
- Added DB engine/session helpers (`db.py`) with `DATABASE_URL` override support.
- Added repository helpers for create/get/update patterns.
- Added DB unit tests:
  - schema create + insert + query roundtrip
  - protection against mutable default sharing for JSON fields
  - repository roundtrip test
- Switched timestamps to timezone-aware UTC to avoid deprecated `utcnow()` usage.

**Why**
- Establish a reliable persistence layer before building upload/inference workflows.

**Files touched**
- `services/api/app/models.py`
- `services/api/app/db.py`
- `services/api/app/database.py`
- `services/api/app/repository.py`
- `services/api/tests/test_db_models.py`
- `services/api/tests/test_repository.py`
- `services/api/requirements.txt`

**How tested**
- [x] Unit tests: `py -3 -m pytest -q services/api/tests`
  - Result: `6 passed`

**Notes / Follow-ups**
- Next step: wire DB initialization into FastAPI startup and add upload endpoint (Step B3).

---

### 2026-01-28 12:24 — Workspace re-organization

**Scope:** repo
**Type:** chore

**What changed**
- Moved legacy/hardcoded demo assets into `legacy-demo/`.

**Why**
- Keep repo root clean to build the real product.

**Files touched**
- Many files moved under `legacy-demo/`
- `README.md` updated to reference new paths

**How tested**
- [x] Manual checks: verified key files exist under new paths

**Notes / Follow-ups**
- Consider moving remaining root assets (e.g., `audio*.ogg`, any `.pptx`) into `legacy-demo/` if they are legacy.
