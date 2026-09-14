# RoomSpec AI

> **Handoff guide.** This README is written for the person taking over the project. It covers what the system does, how to get it running on your machine, how to connect the free-tier cloud services, how to deploy, how to operate and extend it, and which parts have and haven't been tested. Read sections 1–4 before your first run; use the rest as reference.

RoomSpec AI turns **a photo of a wall plus a few constraints** (wall width, budget, style) into a **to-scale cabinet layout and a bill of materials (BOM)**. Every part number, dimension and price in the output has been checked against the catalog database.

**Features at a glance**

- Photo analysis (floor line, wall corner, tilt, colour palette) plus CLIP style matching
- Hybrid retrieval: dense + sparse vectors in Qdrant, hard physical constraints in SQL
- Layout solver with two priorities: *fill the wall* or *complete kitchen*
- **Alternatives:** the same wall solved in the next-best finishes, with price deltas
- Grounded LLM installation notes, rejected and replaced if they mention anything unverifiable
- **Saved specs with share links**, a recent-specs list, and a **printable quote / PDF**
- Web UI with a to-scale elevation drawing, BOM export and a compliance panel
- **Optional API keys, rate limiting and CORS allow-list**
- **Catalog admin API** (prices, stock, bulk upsert, reindex) and **Prometheus `/metrics`**

```
Photo + constraints ─► OpenCV ─► CLIP ─► Qdrant (looks right) ─► SQL (physically fits) ─► layout solver ─► grounded BOM ─► Web UI / JSON
```

---

## Contents

1. [Project status at handoff](#1-project-status-at-handoff)
2. [How the system works](#2-how-the-system-works)
3. [Prerequisites](#3-prerequisites)
4. [Local setup (step by step)](#4-local-setup-step-by-step)
5. [Configuration reference](#5-configuration-reference)
6. [Connecting cloud services (free tiers)](#6-connecting-cloud-services-free-tiers)
7. [Running with Docker](#7-running-with-docker)
8. [Putting the code on GitHub and enabling CI/CD](#8-putting-the-code-on-github-and-enabling-cicd)
9. [Deploying](#9-deploying)
10. [Using the app and the API](#10-using-the-app-and-the-api)
11. [Day-to-day operations](#11-day-to-day-operations)
12. [Extending the project](#12-extending-the-project)
13. [Testing, linting and benchmarks](#13-testing-linting-and-benchmarks)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations and suggested next steps](#15-known-limitations-and-suggested-next-steps)
16. [Repository map](#16-repository-map)

---

## 1. Project status at handoff

**Version 1.1.0, feature-complete, not yet deployed.** See [CHANGELOG.md](CHANGELOG.md) for what changed since 1.0.0. Source: <https://github.com/sonusrujan/roomspec-ai> (`main`).

### Tested and working

| Area | How it was checked |
|---|---|
| Test suite | 80 `pytest` tests pass on Python 3.12 and 3.10; `ruff check` and `ruff format --check` are clean |
| Full pipeline with CLIP | Local server; all 5 sample photos run through `/api/v1/spec`; web UI exercised in a browser (upload, samples, elevation drawing, BOM, CSV/JSON export, compliance panel, timings) |
| Post-v1.0 features (local server, CLIP) | Alternatives and "Use this finish"; layout priority; share links (reload restores result and form) and recent list; print layout; API-key prompt in the UI; admin patch/reindex; 429 with `Retry-After`; `/metrics` with templated routes |
| Light install (no torch) | App starts, falls back to hash embeddings automatically, and returns valid BOMs |
| Docker Compose stack | PostgreSQL 16 + Qdrant server + API (light image): health, photo → BOM, search, catalog, payload indexes, restart persistence and `init_db.py` inside the container all verified. The 1.1 additions (saved specs, recent list ordering, admin patch, alternatives) were then re-checked against a PostgreSQL 16 container. Each saved spec is about 6 kB |
| Benchmarks | `scripts/benchmark.py` run with both embedding backends; results in `benchmarks/` and [§13](#13-testing-linting-and-benchmarks) |

### Not tested yet (do these first)

| Item | Why it matters | How to check |
|---|---|---|
| Live LLM calls (Groq / Gemini) | The HTTP request code has only run against mocks, because no API key was available | Set a key ([§6.3](#63-llm-groq-or-google-ai-studio)), call `/api/v1/bom`, confirm `hallucination_check.llm_used` is `true` |
| Neon / Supabase / Qdrant Cloud | Same code paths as the Docker stack, but TLS connection strings and API keys haven't been tried | Follow [§6](#6-connecting-cloud-services-free-tiers), then `/api/v1/health` should report `ok` |
| Render / Hugging Face deploy | Config files are written but have never run | [§8](#8-putting-the-code-on-github-and-enabling-cicd), [§9](#9-deploying) |
| CLIP Docker image (`WITH_CLIP=true`) | Only the light image was built | `docker build -f backend/Dockerfile --build-arg WITH_CLIP=true -t roomspec-ai:clip .` |
| Real photos and real catalogs | The catalog, style labels and sample photos are all synthetic | Collect about 50 labeled real photos before quoting accuracy numbers |

---

## 2. How the system works

### The problem

A cabinet can look right and still fail to install: too deep, no room for the door swing, out of stock, or 4 cm wider than the wall. RoomSpec AI keeps two questions separate:

- **Vector search** decides what *looks* right.
- **Relational constraints** decide what *physically fits*.
- An **LLM** may only describe rows that have already passed both, and its output is checked afterwards.

### Request pipeline (`POST /api/v1/spec`)

| # | Stage | File | What happens |
|---|---|---|---|
| 1 | Gateway | `backend/app/api/v1/routes.py` | Validates constraints JSON (Pydantic), image MIME type (JPEG/PNG/WebP), size (≤ 8 MB) and that the image decodes |
| 2 | CV preprocessing | `services/cv_preprocessor.py` | CLAHE contrast, partial gray-world white balance, Canny + Hough lines → **floor line**, **wall corner**, **camera tilt**; tilt correction; wall mask; k-means dominant colours mapped to catalog finishes in CIELAB |
| 3 | Embeddings | `services/embedding_engine.py` | CLIP ViT-B/32 (512-d) for the photo and query text; dependency-free hash embedder as fallback; sparse lexical vectors for keyword matching |
| 4 | Style retrieval | `services/hybrid_retriever.py` | Three Qdrant searches (dense text, sparse lexical, image), top-40 per category group, merged with **weighted reciprocal-rank fusion (RRF)**; picks one coherent cabinet finish and one countertop material |
| 5 | Hard constraints | `db/postgres.py` | SQL filters: width ≤ wall, depth ≤ limit, price ≤ budget, in stock, finish; plus door-swing clearance |
| 6 | Layout solver | `services/layout_solver.py` | Exact-width knapsack for the base run (at most one sink base), optional tall pantry, wall cabinets, a countertop cut to length, filler panels. Priorities (`fill_width`): countertop › filled width › tall unit › sink › upper cabinets › lowest cost. `complete_kitchen` puts "upper cabinets cover the run" before filled width. Never exceeds the budget |
| 7 | Verification | `layout_solver.compliance_checks` | Re-reads every chosen part from SQL; checks width, budget, stock, depth, clearance, finish consistency and countertop coverage |
| 8 | Grounded BOM | `services/genai_bom.py` | The BOM table is built deterministically from SQL rows. The LLM writes only the summary and install notes; every part id, cm value and $ value it mentions is verified, and any unverifiable output is replaced with a template |
| 9 | Alternatives | `pipeline.solve_alternatives` | Re-solves the wall in the next 3 best-matching finishes (one SQL query + one solver run each, reusing the vector search); each faces the same compliance checks |
| 10 | Response | `services/pipeline.py`, `routes.py` | JSON with BOM, elevation coordinates, compliance checks, alternatives, image analysis, hallucination report and timings. The route saves it to `spec_runs` (shareable by id) and records metrics |

### Design decisions you should know about

These were measured, not guessed. Keep them unless new measurements say otherwise.

- **Fuse ranks, not vectors.** CLIP text-to-text similarity sits around 0.8 while image-to-text sits around 0.25, so averaging the vectors let the text drown out the photo (a walnut room came back as Matte Black). RRF over separate searches fixed it.
- **Give CLIP the whole photo in true colours.** Cropping to the wall made most samples return Charcoal Grey; colour normalization also hurt. CLIP gets the tilt-corrected original padded to a square. The normalized image is used only for geometry.
- **Dense + sparse.** CLIP text search alone scored P@5 0.73, below keyword search's 0.81: it handles paraphrases ("dark noir cupboards") but misses exact catalog words ("shaker", "navy"). Adding the sparse index raised it to 0.83.
- **One Qdrant collection per embedding backend** (`cabinet_modules_clip`, `cabinet_modules_hash`). Both are 512-d but incompatible, so sharing a collection would silently return wrong results.
- **Two kinds of compliance checks.** `constraint` checks (width, budget, depth…) must always pass; a failure marks the plan `infeasible`. `completeness` checks (e.g. no countertop fits a 40 cm depth limit) mark the plan `partial`.

### Runtime modes

The same code runs in every environment; configuration decides the backing services.

| Component | Zero-config default | Production |
|---|---|---|
| Catalog database | SQLite file `data/roomspec.db` | Neon or Supabase PostgreSQL |
| Vector store | Embedded in-memory Qdrant, rebuilt on every start | Qdrant Cloud (persistent) |
| Embeddings | CLIP if torch is installed, otherwise hash | CLIP (or hash on 512 MB hosts) |
| Install notes | Deterministic template | Groq (Llama 3.3 70B) or Gemini 2.5 Flash-Lite, verified |

---

## 3. Prerequisites

| Tool | Version | Needed for | Check |
|---|---|---|---|
| Python | 3.10–3.12 (3.12 recommended) | Everything | `python3 --version` |
| pip | recent | Installing dependencies | `python3 -m pip --version` |
| git | any | Version control | `git --version` |
| Docker Desktop (with Compose v2) | 24+ | Optional: full local stack, image builds | `docker compose version` |
| Disk space | ~3 GB free | torch (~700 MB) + CLIP weights (~600 MB) + images | |
| RAM | 2 GB+ for CLIP, 512 MB for the light build | | |

Accounts, all optional and all free tier: **Neon** or **Supabase**, **Qdrant Cloud**, **GroqCloud** or **Google AI Studio**, **GitHub**, **Render** or **Hugging Face**.

---

## 4. Local setup (step by step)

### Step 1: Get the code

If you received the folder directly:

```bash
cd "path/to/roomspec-ai"
git log --oneline      # should show the v1.0.0 commit
```

From GitHub:

```bash
git clone https://github.com/sonusrujan/roomspec-ai.git
cd roomspec-ai
```

> **Keep the project on an internal disk.** Python virtual environments can't live on exFAT/FAT external drives, and macOS may block some processes from reading removable volumes. See [§14](#14-troubleshooting).

### Step 2: Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies (pick one)

**Option A: full install with CLIP (recommended for development)**

```bash
pip install --upgrade pip
pip install -r backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Option B: light install, no torch (fast, low memory, hash embeddings)**

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Plus dev tools** (tests and linter), with either option:

```bash
pip install -r requirements-dev.txt
```

### Step 4 (optional): Create your `.env`

Nothing is required for a local run. To configure services, copy the template:

```bash
cp .env.example .env
```

Every variable is described in [§5](#5-configuration-reference).

### Step 5: Start the server

```bash
uvicorn app.main:app --app-dir backend --reload --port 8000
```

On first start the app will:

1. Create the SQLite database at `data/roomspec.db` and load the 176-part catalog.
2. With Option A, download the CLIP weights (~600 MB, one time only) to the Hugging Face cache (`~/.cache/huggingface`).
3. Embed the catalog and build the in-memory Qdrant index (about 1 s).

Wait for `Application startup complete.` in the log.

### Step 6: Verify it works

```bash
curl -s localhost:8000/api/v1/health
```

Expected: `"status":"ok"`, `"catalog_rows":176`, `"vector_points":176`, `"embedding_backend":"clip"` (Option A) or `"hash"` (Option B).

Then open:

- **Web app:** <http://localhost:8000>. Click a sample room, then **Generate specification**.
- **Swagger docs:** <http://localhost:8000/docs>

### Step 7: Run the tests

```bash
pytest
```

Expected: `80 passed`. Tests are fully offline and use a temporary SQLite database, in-memory Qdrant and hash embeddings, so they never touch your `.env` services.

---

## 5. Configuration reference

All settings live in `backend/app/core/config.py`. Each one can be set as an environment variable (upper-case name) or in `.env` at the repository root. Environment variables override `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | `development` / `test` / `production` (informational) |
| `LOG_LEVEL` | `INFO` | Python log level |
| `DATABASE_URL` | `sqlite:///<repo>/data/roomspec.db` | Any SQLAlchemy URL. `postgres://` and `postgresql://` are rewritten to the psycopg3 driver automatically |
| `QDRANT_URL` | *(empty)* | Empty = embedded in-memory Qdrant. Set it for Qdrant Cloud or a self-hosted server |
| `QDRANT_API_KEY` | *(empty)* | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `cabinet_modules` | Base name; the embedding backend is appended (`cabinet_modules_clip`) |
| `EMBEDDING_BACKEND` | `auto` | `auto` (CLIP if importable, else hash), `clip` (fail if unavailable), `hash` |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | Hugging Face model id |
| `EMBEDDING_DIM` | `512` | Hash embedder size (CLIP reports its own) |
| `LLM_PROVIDER` | `auto` | `auto` (Groq if its key is set, else Gemini, else none), `groq`, `gemini`, `none` |
| `GROQ_API_KEY` | *(empty)* | GroqCloud key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `GEMINI_API_KEY` | *(empty)* | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model id |
| `LLM_TIMEOUT_S` | `20` | LLM request timeout; on timeout the template is used |
| `VECTOR_TOP_K` | `40` | Candidates per Qdrant search |
| `MAX_ALTERNATIVES` | `3` | Alternative finishes solved per request (0–7; 0 disables) |
| `SAVE_SPECS` | `true` | Store every result in `spec_runs` for share links and the recent list. Photos are never stored |
| `API_KEYS` | *(empty)* | Comma-separated keys. When set, `/spec`, `/bom` and `/search` require an `X-API-Key` header |
| `ADMIN_API_KEYS` | *(empty)* | Comma-separated keys for `/api/v1/admin/*`. The admin API is **disabled** (403) while empty |
| `RATE_LIMIT_PER_MINUTE` | `0` | Requests per client per rolling minute on `/spec`, `/bom`, `/search` (per API key, else per IP); 0 = unlimited |
| `CORS_ORIGINS` | `*` | Comma-separated allowed browser origins, e.g. `https://roomspec.example.com` |
| `METRICS_ENABLED` | `true` | Serve Prometheus metrics at `/metrics` |
| `MAX_UPLOAD_MB` | `8` | Maximum image upload size |
| `AUTO_SEED` | `true` | Seed an empty database and sync the vector index on startup |
| `SEED_FILE` | `data/seed_modules.json` | Catalog loaded when seeding |
| `FRONTEND_DIR` | `frontend/` | Static web UI directory |

> **Secrets:** `.env` is git-ignored. Never commit keys. On hosting platforms, set them as secret environment variables.

---

## 6. Connecting cloud services (free tiers)

You can connect these one at a time. After each change, restart the server and check `/api/v1/health`.

### 6.1 PostgreSQL: Neon *or* Supabase

**Neon**

1. Sign up at <https://neon.tech> and create a project (pick the region closest to your host).
2. On the project dashboard, click **Connect** and copy the connection string.
3. In `.env`:
   ```
   DATABASE_URL=postgresql://<user>:<password>@ep-xxxx.<region>.aws.neon.tech/neondb?sslmode=require
   ```

**Supabase**

1. Sign up at <https://supabase.com> and create a project. Save the database password.
2. Click **Connect** and copy a connection string. If your network or host has no IPv6, use the **Session pooler** string, because the direct connection is IPv6-only.
3. In `.env`:
   ```
   DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
   ```

The table and index are created automatically on startup. To create them by hand instead, run `data/catalog_modules.sql` in the provider's SQL editor.

### 6.2 Qdrant Cloud

1. Sign up at <https://cloud.qdrant.io> and create a **Free** cluster.
2. Create an API key for the cluster.
3. Copy the cluster URL (it ends in `:6333`).
4. In `.env`:
   ```
   QDRANT_URL=https://xxxxxxxx-xxxx.<region>.<cloud>.cloud.qdrant.io:6333
   QDRANT_API_KEY=<your key>
   ```

On startup the app creates the collection (named dense cosine vectors plus sparse lexical vectors with IDF) and payload indexes on `part_id`, `category` and `finish_style`.

### 6.3 LLM: Groq *or* Google AI Studio

Without a key the app writes deterministic installation notes, which is fine for demos.

**GroqCloud** (Llama 3.3 70B)

1. Go to <https://console.groq.com> and create a key under **API Keys**.
2. In `.env`: `GROQ_API_KEY=gsk_...`

**Google AI Studio** (Gemini 2.5 Flash-Lite)

1. Go to <https://aistudio.google.com> and click **Get API key**.
2. In `.env`: `GEMINI_API_KEY=...`

Verify:

```bash
curl -s localhost:8000/api/v1/bom -H 'content-type: application/json' \
  -d '{"max_width_cm":240,"budget_usd":2500,"finish_style":"Natural Oak"}' | python3 -m json.tool | grep -A6 hallucination_check
```

`"llm_used": true` means the LLM text passed verification. `"llm_used": false` with a `fallback_reason` means the call failed, or the model mentioned something it couldn't back up, and the template was used.

### 6.4 Seed the remote stores

The app seeds automatically on startup (`AUTO_SEED=true`). You can also do it explicitly, which is useful from CI or a laptop:

```bash
python data/init_db.py            # seed if empty, re-index if counts differ
python data/init_db.py --force    # upsert every seed row and rebuild vectors
```

`init_db.py` reads the same `.env`. It indexes with whichever embedding backend is active on the machine running it, so run it where `EMBEDDING_BACKEND` matches your deployment.

---

## 7. Running with Docker

Start Docker Desktop first.

### Full stack: PostgreSQL + Qdrant + API

```bash
# Light image with hash embeddings: fast to build, the configuration that was verified
WITH_CLIP=false docker compose up --build

# CLIP image: ~2 GB, slower build, needs >1 GB RAM for the container
docker compose up --build
```

- App: <http://localhost:8000>
- Qdrant dashboard: <http://localhost:6333/dashboard>
- Postgres: `localhost:5432`, user, password and database all `roomspec`

Compose overrides `DATABASE_URL` and `QDRANT_URL` to point at its own containers. Other variables, such as LLM keys, come from `.env` if the file exists.

Useful commands:

```bash
docker compose ps
docker compose logs -f api
docker compose exec api python data/init_db.py --force
docker compose exec postgres psql -U roomspec -c "select count(*) from cabinet_modules"
docker compose down        # stop
docker compose down -v     # stop and delete database and vector volumes
```

### Image only

```bash
docker build -f backend/Dockerfile -t roomspec-ai .                          # light (default)
docker build -f backend/Dockerfile --build-arg WITH_CLIP=true -t roomspec-ai:clip .
docker run --rm -p 8000:8000 --env-file .env roomspec-ai
```

The image runs as a non-root user, listens on `$PORT` (default 8000), and has a health check on `/api/v1/health`. It is built from the **repository root**, not `backend/`.

---

## 8. Putting the code on GitHub and enabling CI/CD

The project already lives at <https://github.com/sonusrujan/roomspec-ai>. To move it to your own account or organisation instead:

1. Create an **empty** repository on GitHub (no README or licence, to avoid merge conflicts), or use **Settings → Transfer ownership** on the existing one.
2. Point your clone at it and push:
   ```bash
   git remote set-url origin https://github.com/<you>/roomspec-ai.git
   git push -u origin main
   ```
3. CI runs on every push to `main` and on pull requests (`.github/workflows/ci-cd.yml`):
   - **lint-test:** `ruff check`, `ruff format --check`, `pytest` on Python 3.10 and 3.12
   - **docker:** builds the light image and smoke-tests `/health` and `/bom` inside the container
   - **deploy:** on pushes to `main` only, calls the Render deploy hook if the secret exists
4. To enable auto-deploy, add a repository secret under **Settings → Secrets and variables → Actions**:
   - `RENDER_DEPLOY_HOOK_URL`: see [§9.1](#91-render-free-tier-512-mb-ram)

---

## 9. Deploying

### 9.1 Render (free tier, 512 MB RAM)

CLIP and torch don't fit in 512 MB, so Render runs **hash embeddings**. `render.yaml` is already set up for that.

1. Push to GitHub ([§8](#8-putting-the-code-on-github-and-enabling-cicd)).
2. In Render, click **New → Blueprint** and select the repository; Render reads `render.yaml`.
3. When prompted, fill in the secret variables:
   - `DATABASE_URL`: Neon or Supabase. Strongly recommended: without it the app uses SQLite inside the container, which is wiped on every deploy (it re-seeds itself, but data changes are lost).
   - `QDRANT_URL`, `QDRANT_API_KEY`: optional; without them the index is rebuilt in memory on each start, which is fast for 176 parts.
   - `GROQ_API_KEY`: optional.
4. Deploy, then open `https://<service>.onrender.com/api/v1/health`.
5. For deploys from CI: in Render, go to **Settings → Deploy Hook**, copy the URL, and save it as the GitHub secret `RENDER_DEPLOY_HOOK_URL`.

Free Render services sleep after about 15 minutes idle, so the first request afterwards takes up to a minute.

### 9.2 Hugging Face Spaces (free CPU, 16 GB RAM, CLIP)

*Not yet tested.* The recipe:

1. Create a Space with **SDK: Docker**.
2. Spaces expects a `Dockerfile` at the repository root. Copy it there (`cp backend/Dockerfile Dockerfile`) and change the build arg default to `ARG WITH_CLIP=true`.
3. Spaces routes traffic to port 7860 by default. Either add `app_port: 8000` to the Space's README front-matter, or set the variable `PORT=7860`.
4. Add secrets in Space **Settings**: `DATABASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`.
5. Set `DATABASE_URL` to Neon or Supabase. Spaces may run the container under a different user ID than the image's `roomspec` user, in which case the default SQLite file in `/app/data` isn't writable. A remote database avoids that; `sqlite:////tmp/roomspec.db` also works.

A Render (hash) deployment and a Spaces (CLIP) deployment can share one Qdrant cluster: their collections are named separately.

### 9.3 Production checklist

- [ ] Remote `DATABASE_URL` set; `/health` shows `"database":"postgresql"`
- [ ] Qdrant Cloud connected; `/health` shows `"vector_store":"qdrant-remote"` and `vector_points == catalog_rows`
- [ ] LLM key set and one `/bom` call returns `"llm_used": true`
- [ ] `CORS_ORIGINS` set to your real domain(s) instead of `*`
- [ ] `API_KEYS` set if the API should not be public, and `RATE_LIMIT_PER_MINUTE` set (e.g. `30`)
- [ ] `ADMIN_API_KEYS` set to a long random value (`python -c "import secrets; print(secrets.token_urlsafe(32))"`) if you need the admin API
- [ ] `/metrics` scraped by your monitoring, or `METRICS_ENABLED=false`
- [ ] `LOG_LEVEL=INFO` or `WARNING`; logs contain no keys

---

## 10. Using the app and the API

### Web UI (`/`)

1. **Room photo** (optional): upload or drop a JPEG/PNG/WebP, or click a sample room. After generating, the detected floor line (dashed orange) and wall corner (dashed white) are drawn on the preview.
2. **Wall & budget:** wall run width (cm), budget ($), cabinet finish (or let the photo and style decide), free-text style intent.
   **When the budget can't cover everything:** *Fill the wall* (use the full width, add uppers with what's left) or *Complete kitchen* (a shorter run that includes upper cabinets).
3. **Site constraints** (expand): maximum depth, ceiling height, front clearance for door swing; toggles for wall cabinets, countertop and tall pantry.
4. **Generate specification.** The results panel shows:
   - status (**Fits** / **Partial fit** / **No fit**), total and budget use, **Copy share link** and **Print / PDF**
   - **Alternatives:** the same wall in up to 3 other finishes, with total, difference from the current plan, run width and style match; **Use this finish** re-runs with it
   - a to-scale **front elevation** with a dimension chain
   - the **BOM** table, with **Export CSV** and **Copy JSON**
   - **installation notes**, **compliance checks** and the grounding report
   - **photo analysis** and **pipeline timings** (screen only; not printed)
5. **Sharing:** after each run the address bar becomes `/?spec=<id>`. Opening that link restores the result and refills the form. The start page lists the most recent specifications.
6. **Printing:** *Print / PDF* opens the browser print dialog with a quote layout (header with id, date, constraints and share link; no form or buttons). Choose "Save as PDF" to get a file.
7. **API key:** if the server has `API_KEYS` set, the first request shows an "API key required" field. The key is stored in that browser's localStorage.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/spec` | Multipart: `constraints` (JSON string) + optional `image`. Full pipeline |
| `POST` | `/api/v1/bom` | JSON `DesignConstraints` only, no photo |
| `POST` | `/api/v1/search` | Hybrid free-text catalog search with hard filters |
| `GET` | `/api/v1/catalog` | SQL filter: `category` (repeatable), `finish_style`, `max_width_cm`, `max_depth_cm`, `max_price_usd`, `in_stock_only`, `limit` |
| `GET` | `/api/v1/catalog/facets` | Categories, all finishes, cabinet finishes |
| `GET` | `/api/v1/catalog/{part_id}` | One part |
| `GET` | `/api/v1/catalog/{part_id}/thumbnail.svg` | Generated to-scale front drawing |
| `GET` | `/api/v1/samples` | Bundled sample photos |
| `GET` | `/api/v1/health` | Database/vector counts, embedding backend, LLM provider |
| `GET` | `/api/v1/specs?limit=20` | Recent saved specifications (summaries) |
| `GET` | `/api/v1/specs/{request_id}` | A saved specification (full response) |
| `PATCH` | `/api/v1/admin/catalog/{part_id}` | **Admin.** Change `price_usd` and/or `in_stock`; no reindex needed |
| `PUT` | `/api/v1/admin/catalog` | **Admin.** Bulk upsert of catalog rows (up to 5000); re-embeds them |
| `POST` | `/api/v1/admin/reindex` | **Admin.** Rebuild every vector from SQL |
| `GET` | `/metrics` | Prometheus text format (not in Swagger) |

Endpoints marked **Admin** need `X-API-Key: <one of ADMIN_API_KEYS>`. With `API_KEYS` set, `POST /spec`, `/bom` and `/search` need `X-API-Key: <one of API_KEYS>`. Read-only endpoints stay public; saved specs are reachable by their unguessable 16-hex-character id.

**`DesignConstraints` fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_width_cm` | number | 240 | 20 < w ≤ 1200 |
| `budget_usd` | number | 1500 | > 0 |
| `finish_style` | string | null | e.g. `"Matte Walnut"` (case-insensitive) |
| `style_prompt` | string | null | e.g. `"warm mid-century wood"` |
| `max_depth_cm` | number | null | Excludes deeper parts, including 62 cm countertops |
| `ceiling_height_cm` | number | null | Excludes wall and tall units that won't fit |
| `front_clearance_cm` | number | null | Excludes parts whose door/drawer swing needs more space |
| `include_wall_cabinets` | bool | true | |
| `include_countertop` | bool | true | |
| `include_tall_units` | bool | false | |
| `layout_priority` | `"fill_width"` \| `"complete_kitchen"` | `"fill_width"` | What to sacrifice first when the budget can't cover the full width *and* upper cabinets |

Unknown fields are rejected with `422`.

**Examples**

```bash
# Photo + constraints
curl -s localhost:8000/api/v1/spec \
  -F 'constraints={"max_width_cm":240,"budget_usd":1500,"style_prompt":"warm walnut","front_clearance_cm":50}' \
  -F 'image=@data/samples/warm_walnut_kitchen.jpg;type=image/jpeg'

# Constraints only
curl -s localhost:8000/api/v1/bom -H 'content-type: application/json' \
  -d '{"max_width_cm":300,"budget_usd":3500,"finish_style":"Sage Green","include_tall_units":true,"ceiling_height_cm":250}'

# Catalog search
curl -s localhost:8000/api/v1/search -H 'content-type: application/json' \
  -d '{"query":"shaker farmhouse cupboards","max_width_cm":60,"top_k":5}'
```

**Response essentials** (`/spec` and `/bom`): `request_id`, `status`, `total_usd`, `run_width_cm`, `wall_gap_cm`, `finish_style_resolved`, `finish_style_score`, `alternatives[]` (finish, total, run width, status, style score, part ids), `bom[]`, `layout[]` (elevation coordinates per unit), `compliance[]` (each with `kind` of `constraint` or `completeness`), `summary`, `installation_notes[]`, `hallucination_check`, `image_analysis`, `timings_ms`.

**Error codes:** `401` missing/invalid API key · `403` admin API disabled · `404` unknown part or spec · `413` image too large · `415` unsupported image type · `422` invalid input, or an image that won't decode · `429` rate limit exceeded (see the `Retry-After` header).

**Admin examples**

```bash
export ADMIN="X-API-Key: <admin key>"

# Price change / out of stock
curl -s -X PATCH localhost:8000/api/v1/admin/catalog/RS-BC-060-MWAL -H "$ADMIN" \
  -H 'content-type: application/json' -d '{"price_usd": 249.00, "in_stock": false}'

# Add or update parts (image_url is optional)
curl -s -X PUT localhost:8000/api/v1/admin/catalog -H "$ADMIN" -H 'content-type: application/json' -d '[
  {"part_id":"ACME-WC-045-TERRA","part_name":"Terracotta Wall Cabinet 45 cm","category":"Wall Cabinet",
   "finish_style":"Terracotta Clay","price_usd":199,"width_cm":45,"height_cm":72,"depth_cm":33,
   "door_clearance_cm":35,"description":"burnt orange painted wall unit"}]'

# Full rebuild after SQL edits made outside the API
curl -s -X POST localhost:8000/api/v1/admin/reindex -H "$ADMIN"
```

---

## 11. Day-to-day operations

| Task | Command / action |
|---|---|
| Check health | `curl <host>/api/v1/health`. `degraded` means the database is unreachable or row count ≠ vector count |
| Re-seed and re-index | `python data/init_db.py --force` |
| Regenerate the synthetic catalog | `python scripts/generate_seed.py`, then `python data/init_db.py --force` |
| Regenerate sample photos | `python scripts/make_samples.py` |
| Re-run benchmarks | `python scripts/benchmark.py --backend clip` (or `hash`); writes `benchmarks/RESULTS-*.md` |
| Switch embedding backend | Set `EMBEDDING_BACKEND`, restart; a separate collection is built automatically |
| Inspect the database | `sqlite3 data/roomspec.db "select part_id, price_usd, in_stock from cabinet_modules limit 10"`, or your Postgres console |
| Inspect vectors | Qdrant dashboard (`:6333/dashboard` locally, or the cloud console) |
| Monitor | Scrape `/metrics`: `roomspec_http_requests_total`, `roomspec_http_request_duration_seconds`, `roomspec_pipeline_stage_seconds{stage=…}`, `roomspec_specs_total{status=…}`, `roomspec_llm_narratives_total{provider,outcome}` |
| Find a user's result | `GET /api/v1/specs/<id>` (the id is in their share link), or `select payload from spec_runs where request_id = '<id>'` |
| Prune old saved specs | `delete from spec_runs where created_at < now() - interval '90 days'` (there's no automatic retention yet) |

**Changing a price or stock level:** use `PATCH /api/v1/admin/catalog/{part_id}` ([§10](#10-using-the-app-and-the-api)) or update the row in PostgreSQL. Prices, dimensions and stock are read from SQL on every request, so no re-index is needed. Re-index only when names, descriptions, finishes or categories change, because those feed the embeddings.

---

## 12. Extending the project

### Load a real catalog

1. Produce a JSON array with the same fields as `data/seed_modules.json`:
   `part_id, part_name, category, finish_style, material, price_usd, width_cm, height_cm, depth_cm, door_clearance_cm, in_stock, description, image_url`
2. Keep the category names the solver knows: `Base Cabinet`, `Drawer Base`, `Sink Base`, `Wall Cabinet`, `Tall Pantry`, `Filler Panel`, `Countertop` (`Corner Base Cabinet` is stored but not yet used by the solver). They are listed in `services/hybrid_retriever.py`.
3. Either send it to `PUT /api/v1/admin/catalog` (validated and re-embedded automatically), or point `SEED_FILE` at the file and run `python data/init_db.py --force`.
4. Add each new finish's display colour to `backend/app/core/finishes.py`. It drives photo colour matching, thumbnails and the elevation drawing.
5. The grounding verifier recognises part ids shaped like `AB-XX-123` (`PART_ID_RE` in `services/genai_bom.py`). If your part numbers look different, update that regex or unknown ids won't be caught.
6. Re-run the benchmarks and the tests. `tests/test_layout_solver.py` reads the seed file directly, so a real catalog may need new test fixtures.

### Tune style matching

- Fusion weights: `build_style_query` in `services/hybrid_retriever.py` (user text vs photo).
- Catalog text that gets embedded: `module_document` (dense) and `lexical_document` (sparse) in `services/embedding_engine.py`.
- Build a real labeled evaluation set first; the current one has only 24 queries and 5 synthetic photos.

### Change layout rules

- Constants (`PLINTH_CM`, `BACKSPLASH_GAP_CM`, `MAX_FILLER_GAP_CM`) and the priority key are in `services/layout_solver.py`.
- Add a compliance check in `compliance_checks`, choosing `kind="constraint"` or `"completeness"`, and a test in `tests/test_layout_solver.py`.

### Add an API endpoint

Add a route to `backend/app/api/v1/routes.py` with Pydantic models in `backend/app/schemas/spec.py`, plus a contract test in `tests/test_api_routes.py`. It appears in `/docs` automatically.

---

## 13. Testing, linting and benchmarks

```bash
pytest                         # 80 tests, offline, ~2 s
pytest tests/test_layout_solver.py -q
ruff check .                   # lint
ruff format .                  # format (CI runs --check)
```

| Test file | Covers |
|---|---|
| `tests/test_cv_pipeline.py` | Upload integrity, CLAHE, white balance, floor/corner/tilt detection, finish suggestions |
| `tests/test_layout_solver.py` | Width/budget never exceeded (18-case matrix), exact fills, countertop cut, ceiling, tall unit, tamper detection |
| `tests/test_genai_bom.py` | BOM aggregation, verifier accepts grounded text and catches invented parts/sizes/prices, LLM fallback paths |
| `tests/test_api_routes.py` | Every endpoint, validation errors, collection naming, end-to-end grounding against SQL |
| `tests/test_features.py` | Layout priority, alternatives, saved specs, API keys, rate limiter (unit + 429), admin patch/upsert/reindex and validation, metrics labels |

### Benchmark results (measured)

Reproduce with `python scripts/benchmark.py --backend clip|hash`. Measured on an Apple Silicon Mac, CPU only, embedded Qdrant + SQLite (no network latency) and no LLM call.

<!-- BENCHMARKS:START -->
**CLIP ViT-B/32 backend** (`benchmarks/RESULTS-clip.md`)

| Metric | Keyword baseline | RoomSpec AI | Method |
|---|---|---|---|
| Dimension compliance (whole recommendation) | 24.0% | 100.0% | 100 random wall/budget/depth/clearance scenarios |
| Dimension compliance (per part) | 63.8% | 100.0% | Same scenarios, each recommended part checked |
| Style Precision@5 (text) | 0.81 | 0.83 | 24 labeled finish queries, mostly paraphrased |
| Finish accuracy from photo | n/a | 80.0% | 5 synthetic sample rooms |
| CLIP image embedding | n/a | 26.8 ms P95 | ViT-B/32 forward pass, CPU, batch 1 (P50 21.07 ms) |
| OpenCV preprocessing | n/a | 43.05 ms P95 | 960×720 photo (P50 40.8 ms) |
| Vector + SQL retrieval | n/a | 16.86 ms P95 | Qdrant (embedded) + SQL filters (P50 15.94 ms) |
| End-to-end `/api/v1/spec` | n/a | 103.46 ms P95 | Photo upload → BOM, in-process client (P50 90.85 ms) |
| BOM lines not matching SQL | n/a | 0 | Every line re-checked against the catalog row |
| Narratives failing grounding check | n/a | 0 / 100 | Part ids, cm and $ values verified |

**Hash backend, no torch** (`benchmarks/RESULTS-hash.md`)

| Metric | Keyword baseline | RoomSpec AI | Method |
|---|---|---|---|
| Dimension compliance (whole recommendation) | 24.0% | 100.0% | 100 random wall/budget/depth/clearance scenarios |
| Dimension compliance (per part) | 63.8% | 100.0% | Same scenarios, each recommended part checked |
| Style Precision@5 (text) | 0.81 | 0.80 | 24 labeled finish queries, mostly paraphrased |
| Finish accuracy from photo | n/a | 80.0% | 5 synthetic sample rooms |
| OpenCV preprocessing | n/a | 41.11 ms P95 | 960×720 photo (P50 40.25 ms) |
| Vector + SQL retrieval | n/a | 10.79 ms P95 | Qdrant (embedded) + SQL filters (P50 10.55 ms) |
| End-to-end `/api/v1/spec` | n/a | 57.76 ms P95 | Photo upload → BOM, in-process client (P50 56.6 ms) |
| BOM lines not matching SQL | n/a | 0 | Every line re-checked against the catalog row |
| Narratives failing grounding check | n/a | 0 / 100 | Part ids, cm and $ values verified |

End-to-end latency includes solving 3 alternative plans (about 3 ms).
<!-- BENCHMARKS:END -->

**Read these honestly:**
- Catalog, labels and photos are synthetic; real-world accuracy is unmeasured.
- On Qdrant Cloud, Neon and a live LLM, expect tens of milliseconds more per database call and hundreds of milliseconds to seconds for the LLM.
- "Keyword baseline" = top-5 rows by keyword overlap with no dimensional reasoning, the failure mode this project exists to fix.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` on port 8000 | Another server is running | `lsof -i :8000` then stop it, or use `--port 8001` |
| `pip install` fails on torch | Unsupported Python/OS combination, or no CPU wheel | Use Python 3.12, or install the light build (`backend/requirements.txt`) |
| Startup log: `CLIP unavailable … falling back to hash` | torch not installed, or model download blocked | Install Option A, or allow access to `huggingface.co`; pre-download with `python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"` |
| First start is slow | CLIP weight download (~600 MB) | One time only; later starts take ~8 s to load the model |
| `/health` returns `degraded` | Database unreachable, or vector count ≠ row count | Check `DATABASE_URL`; run `python data/init_db.py --force` |
| `connection refused` / timeout to Supabase | Direct connection is IPv6-only | Use the Session pooler connection string |
| `SSL connection is required` | Missing TLS parameter | Add `?sslmode=require` to `DATABASE_URL` |
| Qdrant `401` / `403` | Wrong or missing API key | Check `QDRANT_API_KEY` and that `QDRANT_URL` includes `:6333` |
| `hallucination_check.fallback_reason: LLM call failed: HTTPStatusError` | Invalid key, wrong model id or rate limit | Check the key and `GROQ_MODEL` / `GEMINI_MODEL`; the app keeps working with template notes |
| Every result `infeasible` | Constraints exclude all base units, e.g. `max_depth_cm` < 58 or a tiny budget | Expected behaviour; loosen the constraints |
| `401 Missing or invalid API key` | `API_KEYS` is set on the server | Send `X-API-Key`; in the UI, enter the key in the field that appears |
| `403 Admin API is disabled` | `ADMIN_API_KEYS` is empty | Set it and restart |
| `429 Too Many Requests` | `RATE_LIMIT_PER_MINUTE` exceeded | Wait `Retry-After` seconds, or raise the limit. The limit applies per process, so N workers allow N× the limit |
| Share link says "no longer exists" | Spec was deleted, `SAVE_SPECS=false`, or the server uses a fresh SQLite file (e.g. a redeploy without `DATABASE_URL`) | Use a persistent PostgreSQL `DATABASE_URL` |
| `422` from `/spec` | `constraints` isn't valid JSON, has an unknown field, or the image won't decode | Read `detail` in the response |
| Venv errors or `._*` files appear (macOS) | Project is on an exFAT/FAT external drive | Move the project to an internal disk, or keep the venv elsewhere (e.g. `~/.venvs/roomspec-ai`) and delete junk with `find . -name '._*' -delete` |
| Server or `docker compose build` hangs with no output (macOS, external drive) | macOS is blocking the process from reading a removable volume | Grant access in **System Settings → Privacy & Security → Files and Folders**, run from a regular terminal, or work from a copy on an internal disk |

---

## 15. Known limitations and suggested next steps

In rough priority order:

1. **Access control is key-based only.** Optional shared API keys, an in-process rate limiter and a CORS allow-list now exist, but there are no user accounts, per-user spec ownership or a distributed rate limit. Saved specs are public to anyone holding the id.
2. **Straight runs only.** L- and U-shaped kitchens need corner units (already in the catalog) and multi-wall constraints in the solver.
3. **Photo geometry isn't metric.** Floor line, corner and tilt are estimated, but wall width always comes from the user. A reference object (e.g. an A4 sheet) or device depth data could make it metric.
4. **Small, synthetic evaluation.** Build a real labeled set before tuning fusion weights or colour heuristics.
5. **Budget policy is fixed.** The solver fills width first, then adds upper cabinets with the remaining budget. Some users would prefer a narrower run with uppers; expose this as a parameter.
6. **Countertop joints** are used only when no single piece is long enough, and the joint position isn't optimised (e.g. away from the sink).
7. **Verifier scope.** It checks part ids, cm values and $ values, but not mm, inches or free-text claims such as brand names.
8. **Saved specs have no retention policy** and grow without bound; add a scheduled cleanup (see [§11](#11-day-to-day-operations)). Metrics are also per process.
9. **Embedded Qdrant is in memory**, so it rebuilds on every restart. That's fine for 176 parts, but use Qdrant Cloud or a server for large catalogs.

---

## 16. Repository map

```
.
├── README.md                          ← you are here
├── CHANGELOG.md                       release notes
├── .env.example                       all settings with comments
├── docker-compose.yml                 PostgreSQL 16 + Qdrant + API
├── render.yaml                        Render blueprint (light image)
├── pyproject.toml                     pytest + ruff config
├── requirements-dev.txt               runtime + pytest + ruff
├── .github/workflows/ci-cd.yml        lint → test (3.10, 3.12) → docker smoke test → deploy hook
├── backend/
│   ├── Dockerfile                     build from repo root; WITH_CLIP build arg
│   ├── requirements.txt               core runtime (no torch)
│   ├── requirements-ml.txt            + torch + transformers (CLIP)
│   └── app/
│       ├── main.py                    app, startup bootstrap, static frontend
│       ├── api/v1/routes.py           all endpoints, upload validation, SVG thumbnails
│       ├── core/config.py             settings (env / .env)
│       ├── core/finishes.py           finish colour swatches
│       ├── core/security.py           API keys, admin keys, sliding-window rate limiter
│       ├── core/metrics.py            Prometheus registry (no extra dependency)
│       ├── db/postgres.py             table definition, hard-constraint filters, upsert
│       ├── db/qdrant.py               collection (dense + sparse), upsert, search
│       ├── db/spec_store.py           saved specification runs (share links)
│       ├── db/bootstrap.py            schema + seed + index orchestration
│       ├── schemas/spec.py            request/response models
│       └── services/
│           ├── cv_preprocessor.py     OpenCV pipeline
│           ├── embedding_engine.py    CLIP / hash embedders, sparse lexical vectors
│           ├── hybrid_retriever.py    RRF fusion, finish resolution, SQL filtering
│           ├── layout_solver.py       knapsack layout, compliance checks
│           ├── genai_bom.py           BOM lines, LLM calls, grounding verifier
│           └── pipeline.py            orchestration, alternatives, elevation coordinates
├── frontend/
│   ├── index.html                     Tailwind (CDN) UI
│   └── app.js                         form, fetch, elevation / BOM rendering, CSV export
├── data/
│   ├── seed_modules.json              176-part synthetic catalog
│   ├── catalog_modules.sql            PostgreSQL schema
│   ├── init_db.py                     seed + index script
│   └── samples/                       5 synthetic room photos
├── scripts/
│   ├── generate_seed.py               catalog generator
│   ├── make_samples.py                sample photo renderer
│   └── benchmark.py                   benchmark harness
├── benchmarks/                        measured results (Markdown + JSON)
└── tests/                             80 hermetic pytest tests
```
