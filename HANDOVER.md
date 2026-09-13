# RoomSpec AI: Handover

Status as of 2026-09-13: **feature-complete v1.0.0**. The full pipeline runs locally and in Docker, with 63 passing tests and measured benchmarks. The code has not yet been deployed or connected to live cloud accounts. Read `README.md` for architecture and API; this document covers what the next owner needs to know.

## 1. What has been verified

| Area | How it was verified |
|---|---|
| Unit and API tests | `pytest`: 63 passed on Python 3.12 and 3.10; `ruff check` / `ruff format --check` clean |
| Full pipeline with CLIP | Local server; all 5 sample photos run through `/api/v1/spec`; web UI checked in a browser (upload, samples, elevation, BOM, CSV/JSON export, compliance, timings) |
| Docker stack | `docker compose` with PostgreSQL 16 + Qdrant server + API (light image): see §6 |
| Benchmarks | `scripts/benchmark.py --backend clip` and `--backend hash`; results committed in `benchmarks/` |

## 2. What has NOT been verified

- **Live LLM calls.** The Groq and Gemini request code (`services/genai_bom.py: _call_groq/_call_gemini`) is only exercised with mocks, and no API key was available. Before relying on it, set a key and run one `/api/v1/bom` request; confirm `hallucination_check.llm_used` is `true`.
- **Neon / Supabase / Qdrant Cloud.** The code paths are the same as in the Docker stack, since a Postgres URL and a Qdrant URL are just different hosts, but TLS URLs (`?sslmode=require`) and API keys against the managed services have not been tried.
- **Render / Hugging Face deploy.** `render.yaml` and the CI deploy hook are written but have never run. The CI workflow has not run on GitHub either, because there is no remote yet.
- **Real photos and real catalogs.** All photos, style labels and catalog rows are synthetic, so benchmark precision will not transfer. Collect about 50 real labeled photos before quoting accuracy.

## 3. Getting it running

```bash
# Local, offline
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn app.main:app --app-dir backend --reload      # http://localhost:8000

# Tests
pip install -r requirements-dev.txt && pytest

# Docker (Postgres + Qdrant + API)
WITH_CLIP=false docker compose up --build
```

> **Machine-specific note (original dev box):** the repo lives on an exFAT external drive. Python virtualenvs cannot live on exFAT, so the working venv is `~/.venvs/roomspec-ai`. macOS also blocked some processes (the IDE preview launcher, Docker builds) from reading the drive until they were run from a normal shell or from a copy on the internal disk. None of this applies once the repo is cloned onto a normal disk.

## 4. Going to production (free tier)

1. Create a GitHub repo and push it. CI (`.github/workflows/ci-cd.yml`) runs lint, tests on 3.10 and 3.12, then a Docker build plus container smoke test.
2. **Neon or Supabase:** create a database and copy its connection string into `DATABASE_URL`.
3. **Qdrant Cloud:** create a free cluster, then set `QDRANT_URL` and `QDRANT_API_KEY`.
4. **GroqCloud or Google AI Studio:** set `GROQ_API_KEY` or `GEMINI_API_KEY`.
5. Seed and index the remote stores once: `python data/init_db.py --force`.
6. **Render:** create a Blueprint from `render.yaml`, add the secrets above, and copy the service's deploy hook URL into the GitHub secret `RENDER_DEPLOY_HOOK_URL`. Pushes to `main` then deploy automatically.
   - Render free has 512 MB of RAM, so it runs hash embeddings. For CLIP, deploy the image built with `WITH_CLIP=true` to a Hugging Face Space (Docker SDK, CPU basic).
   - The Qdrant collection name is suffixed with the embedding backend (`cabinet_modules_hash`, `cabinet_modules_clip`). Both are 512-d but live in unrelated vector spaces, so a Render (hash) and a Hugging Face (CLIP) deployment can share one cluster safely.

## 5. Operating it

| Task | How |
|---|---|
| Change catalog data | Edit `scripts/generate_seed.py` (or replace `data/seed_modules.json` with real data using the same fields), then `python data/init_db.py --force` |
| Rebuild vectors only | `python data/init_db.py --force`; embedded Qdrant rebuilds automatically on every start |
| Check health | `GET /api/v1/health` returns `degraded` if the DB row count ≠ vector count, or the DB is unreachable |
| Re-measure | `python scripts/benchmark.py --backend clip`, then paste the tables into README |
| Regenerate sample photos | `python scripts/make_samples.py` |

Configuration lives in `backend/app/core/config.py`, and every field can be overridden with an environment variable of the same name in upper case (see `.env.example`).

## 6. Docker smoke test

<!-- DOCKER:START -->
Run on 2026-09-13 with `WITH_CLIP=false docker compose up --build` (Docker 29.3, Compose v5.1). The image is 663 MB.

| Check | Result |
|---|---|
| `/api/v1/health` | `ok`, database `postgresql`, 176 rows = 176 vectors in the Qdrant server |
| `/api/v1/spec` with the sage sample photo, 240 cm, $2,000, 50 cm clearance, 250 cm ceiling | `ok`, Sage Green, $1,755.20, 240 cm run, all constraint checks pass, grounding check passes |
| `/api/v1/search` "shaker farmhouse", width ≤ 60 | top 3 all Cream Shaker |
| Qdrant collection `cabinet_modules_hash` | named `dense` + sparse `lexical` vectors; payload indexes on `category`, `finish_style`, `part_id` |
| `docker restart` of the API | bootstrap reused existing data (`seeded: 0, indexed: 0`) |
| `python data/init_db.py` inside the container | idempotent (`seeded: 0, indexed: 0`) |
| PostgreSQL | `cabinet_modules`: 176 rows, 150 in stock |

The CLIP image variant (`WITH_CLIP=true`) was not built. Its code path is the same one verified locally with CLIP.
<!-- DOCKER:END -->

## 7. Known limitations and suggested next steps

In rough priority order:

1. **Straight runs only.** The solver handles a single wall. L-shaped and U-shaped kitchens need corner units (`Corner Base Cabinet` is already in the catalog but unused) and multi-wall constraints.
2. **Photo geometry is descriptive, not metric.** Floor line, corner and tilt are estimated, but the photo never provides the wall width; the user's measurement is the source of truth. A reference object (for example an A4 sheet) or ARKit depth data could make the estimate metric.
3. **Style evaluation is small and synthetic.** 24 text queries and 5 generated photos. Build a real labeled set before tuning the RRF weights (`hybrid_retriever.build_style_query`) or the colour heuristics.
4. **Budget policy.** The solver fills width first, then adds wall cabinets with the remaining money. Some users would rather have a narrower run that includes uppers; that preference could be exposed as a parameter.
5. **Countertop joints.** Two pieces are joined only when no single piece is long enough, and the joint position is not optimised (for example, kept away from the sink).
6. **No auth or rate limiting.** CORS is `*` and there is no authentication. Add both before exposing uploads publicly; the upload size limit is 8 MB (`MAX_UPLOAD_MB`).
7. **LLM prompt.** Kept deliberately small. The verifier checks part ids, cm and $ values but not other units (mm, inches) or free-text claims such as brand names.

## 8. Code map

```
backend/app/main.py                  app, lifespan bootstrap (seed + index), static frontend
backend/app/api/v1/routes.py         all HTTP endpoints, upload validation, SVG thumbnails
backend/app/core/config.py           settings (env / .env)
backend/app/core/finishes.py         finish colour swatches (CV matching, thumbnails, elevation)
backend/app/db/postgres.py           SQLAlchemy table, hard-constraint filters, upsert
backend/app/db/qdrant.py             collection (dense + sparse), upsert, search
backend/app/db/bootstrap.py          schema + seed + index orchestration
backend/app/services/cv_preprocessor.py   OpenCV pipeline
backend/app/services/embedding_engine.py  CLIP / hash embedders, lexical sparse vectors
backend/app/services/hybrid_retriever.py  RRF fusion, finish resolution, SQL filtering
backend/app/services/layout_solver.py     knapsack layout, compliance checks
backend/app/services/genai_bom.py         BOM lines, fact sheet, LLM calls, grounding verifier
backend/app/services/pipeline.py          end-to-end orchestration, elevation coordinates
frontend/                            Tailwind CDN + vanilla JS UI
scripts/                             seed generator, sample renderer, benchmark harness
tests/                               pytest (hermetic: SQLite, in-memory Qdrant, hash, no LLM)
```
