# RoomSpec AI

**Computer vision and hybrid retrieval for interior specs.** Upload a photo of a wall and give its width, budget and style. RoomSpec AI returns a to-scale cabinet layout and a bill of materials (BOM). Every part number, dimension and price in it is checked against the catalog database.

```
Photo + constraints ─► OpenCV ─► CLIP ─► Qdrant (style) ─► SQL (physics) ─► layout solver ─► grounded BOM ─► Web UI / JSON
```

A cabinet can look right and still fail to install: too deep, no room for the door swing, out of stock, or it overruns the wall by 4 cm. RoomSpec AI splits the problem in two. Vector search decides **what looks right**. Relational constraints decide **what physically fits**. An LLM may only describe rows that have already passed both.

---

## Quick start (zero config, fully offline)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu   # or backend/requirements.txt for the light build
uvicorn app.main:app --app-dir backend --reload
```

Open <http://localhost:8000> for the web app or <http://localhost:8000/docs> for Swagger.

With no `.env`, the app uses local SQLite, embedded in-memory Qdrant and deterministic installation notes. On startup it seeds and indexes the 176-part catalog. If `torch` is not installed it automatically falls back to a dependency-free hash embedder.

### Full stack with Docker

```bash
docker compose up --build                  # PostgreSQL 16 + Qdrant + API with CLIP
WITH_CLIP=false docker compose up --build  # lighter image, hash embeddings
```

### Free-tier cloud services

Copy `.env.example` to `.env` and fill in whichever services you use:

| Setting | Service | Notes |
|---|---|---|
| `DATABASE_URL` | Neon or Supabase Postgres | `postgres://…` URLs are rewritten to the psycopg3 driver |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud | Collection, cosine HNSW index and payload indexes are created automatically |
| `GROQ_API_KEY` or `GEMINI_API_KEY` | GroqCloud (Llama 3.3 70B) / Google AI Studio (Gemini 2.5 Flash-Lite) | Optional; without a key, template notes are used |

Then seed and index the remote stores once:

```bash
python data/init_db.py --force
```

---

## How it works

| Stage | File | What it does |
|---|---|---|
| Gateway | `backend/app/api/v1/routes.py` | Pydantic validation, MIME/size/decode integrity checks, 415/413/422 errors |
| CV preprocessing | `services/cv_preprocessor.py` | CLAHE on LAB lightness, partial gray-world white balance, Canny + probabilistic Hough to find the **floor line**, **wall corner** and **camera tilt**, tilt correction, wall ROI mask, k-means dominant colours → nearest catalog finishes in CIELAB |
| Embeddings | `services/embedding_engine.py` | CLIP ViT-B/32 (512-d) for photos and caption-style catalog documents, with a hashing fallback |
| Style retrieval | `services/hybrid_retriever.py` | Three Qdrant queries (dense CLIP text, sparse lexical with server-side IDF, CLIP image), top-40 per category group, merged with **weighted reciprocal-rank fusion** |
| Hard constraints | `db/postgres.py` | SQL filters on width, depth, price, stock and finish, plus door-swing clearance, applied to the vector candidates |
| Layout solver | `services/layout_solver.py` | Exact-width unbounded knapsack for the base run (at most one sink base), optional tall pantry, wall cabinets, a countertop cut to length and filler panels. Objective, in order: countertop coverage › filled width › tall unit › sink › upper-run coverage › lowest cost, never above budget |
| Verification | `layout_solver.compliance_checks` | Re-reads every chosen part from SQL and checks width, budget, stock, depth, clearance and finish consistency |
| Grounded BOM | `services/genai_bom.py` | BOM table built deterministically from SQL rows. The LLM writes only the summary and installation notes from a JSON fact sheet; every part id, cm value and $ value in its output is verified, and any failure is replaced with a template |

### Design decisions worth knowing

- **Fuse ranks, not vectors.** CLIP text-to-text cosine similarity sits near 0.8, while image-to-text sits near 0.25. Averaging the two vectors let the colour-hint text drown out the photo: the walnut sample room resolved to *Matte Black*. Reciprocal-rank fusion of two separate searches fixed it.
- **Give CLIP the whole photo, in true colour.** Cropping to the wall ROI made every sample return *Charcoal Grey*, and colour normalization cost precision too. CLIP gets the tilt-corrected original padded to a square. The normalized image is used only for geometry.
- **Dense plus sparse.** On its own, CLIP text search scored P@5 0.73, below the keyword baseline's 0.81. It handles paraphrases ("dark noir cupboards") but misses exact catalog words ("shaker", "navy"). Adding a BM25-style sparse index and fusing ranks raised it to 0.83.
- **Caption-style catalog text.** Indexing `"a photo of a matte walnut kitchen base cabinet, …"` instead of spec-sheet text raised CLIP text P@5 from 0.76 to 0.90 in an early sweep, on an easier query set than the benchmark's.
- **Constraint vs completeness checks.** "Wider than the wall" is a hard violation. "No countertop fits a 40 cm depth limit" means a requested element is missing: the plan is marked `partial`, never presented as a fit.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/spec` | Multipart: `image` (optional) + `constraints` JSON. Runs the full pipeline |
| `POST` | `/api/v1/bom` | JSON constraints only, with no photo |
| `POST` | `/api/v1/search` | Hybrid free-text catalog search with hard filters |
| `GET` | `/api/v1/catalog` | Pure SQL filter (`category`, `finish_style`, `max_width_cm`, …) |
| `GET` | `/api/v1/catalog/{part_id}` · `/thumbnail.svg` | Part detail and generated to-scale front drawing |
| `GET` | `/api/v1/catalog/facets` | Categories and finishes |
| `GET` | `/api/v1/samples` | Bundled sample room photos |
| `GET` | `/api/v1/health` | DB and vector counts, embedding backend, LLM provider |

```bash
curl -s localhost:8000/api/v1/spec \
  -F 'constraints={"max_width_cm":240,"budget_usd":1500,"style_prompt":"warm walnut","front_clearance_cm":50}' \
  -F 'image=@data/samples/warm_walnut_kitchen.jpg;type=image/jpeg' | jq '.status, .total_usd, .bom[].part_id'
```

`DesignConstraints` fields: `max_width_cm`, `budget_usd`, `finish_style`, `style_prompt`, `max_depth_cm`, `ceiling_height_cm`, `front_clearance_cm`, `include_wall_cabinets`, `include_countertop`, `include_tall_units`.

---

## Data model

`data/catalog_modules.sql` extends the original schema with `material`, `door_clearance_cm` (swing or drawer extension needed in front of the unit) and `description`. The Qdrant collection `cabinet_modules_<backend>` (e.g. `cabinet_modules_clip`) has a named 512-d cosine `dense` vector and a sparse `lexical` vector (IDF modifier), with keyword payload indexes on `part_id`, `category` and `finish_style`.

`data/seed_modules.json` holds 176 parts: 8 cabinet finishes × 7 cabinet types (base, drawer, sink, blind corner, wall, tall pantry, filler) plus 4 countertop materials × 4 lengths. It is generated deterministically by `scripts/generate_seed.py`. Sample room photos are rendered by `scripts/make_samples.py`.

---

## Benchmarks

These are **measured**, not projected. Run `python scripts/benchmark.py --backend clip` (or `hash`) to reproduce; results land in `benchmarks/`. The numbers below come from an Apple Silicon Mac (CPU only) with embedded Qdrant and SQLite, so they include no network latency and no LLM call.

<!-- BENCHMARKS:START -->
**CLIP ViT-B/32 backend** (`benchmarks/RESULTS-clip.md`)

| Metric | Keyword baseline | RoomSpec AI | Method |
|---|---|---|---|
| Dimension compliance (whole recommendation) | 24.0% | 100.0% | 100 random wall/budget/depth/clearance scenarios |
| Dimension compliance (per part) | 63.8% | 100.0% | Same scenarios, each recommended part checked |
| Style Precision@5 (text) | 0.81 | 0.83 | 24 labeled finish queries, mostly paraphrased |
| Finish accuracy from photo | n/a | 80.0% | 5 synthetic sample rooms |
| CLIP image embedding | n/a | 22.48 ms P95 | ViT-B/32 forward pass, CPU, batch 1 (P50 22.19 ms) |
| OpenCV preprocessing | n/a | 42.1 ms P95 | 960×720 photo (P50 41.2 ms) |
| Vector + SQL retrieval | n/a | 16.7 ms P95 | Qdrant (embedded) + SQL filters (P50 16.53 ms) |
| End-to-end `/api/v1/spec` | n/a | 90.99 ms P95 | Photo upload → BOM, in-process client (P50 90.13 ms) |
| BOM lines not matching SQL | n/a | 0 | Every line re-checked against the catalog row |
| Narratives failing grounding check | n/a | 0 / 100 | Part ids, cm and $ values verified |

<details><summary>Hash-embedding backend (no torch, fits 512 MB hosts)</summary>

| Metric | Keyword baseline | RoomSpec AI | Method |
|---|---|---|---|
| Dimension compliance (whole recommendation) | 24.0% | 100.0% | 100 random wall/budget/depth/clearance scenarios |
| Dimension compliance (per part) | 63.8% | 100.0% | Same scenarios, each recommended part checked |
| Style Precision@5 (text) | 0.81 | 0.80 | 24 labeled finish queries, mostly paraphrased |
| Finish accuracy from photo | n/a | 80.0% | 5 synthetic sample rooms |
| OpenCV preprocessing | n/a | 42.05 ms P95 | 960×720 photo (P50 41.26 ms) |
| Vector + SQL retrieval | n/a | 11.31 ms P95 | Qdrant (embedded) + SQL filters (P50 11.06 ms) |
| End-to-end `/api/v1/spec` | n/a | 55.18 ms P95 | Photo upload → BOM, in-process client (P50 54.48 ms) |
| BOM lines not matching SQL | n/a | 0 | Every line re-checked against the catalog row |
| Narratives failing grounding check | n/a | 0 / 100 | Part ids, cm and $ values verified |

</details>
<!-- BENCHMARKS:END -->

Caveats, stated plainly:
- The catalog, style labels and sample photos are synthetic. Precision on real photos and real catalogs will be different, and should be measured before any claims are made.
- Retrieval latency here excludes network round-trips. On Qdrant Cloud and Neon, expect tens of milliseconds more per call, and LLM generation adds hundreds of milliseconds to seconds.
- "Keyword baseline" means the top-5 catalog rows by keyword overlap, with no dimensional reasoning. That is the failure mode this project exists to fix.

---

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest              # 63 tests: CV, layout solver, grounding verifier, API contracts (hermetic, offline)
ruff check . && ruff format --check .
```

`.github/workflows/ci-cd.yml` runs lint and tests on Python 3.10 and 3.12, builds the Docker image, smoke-tests `/health` and `/bom` inside the container, and triggers a Render deploy hook on `main` when `RENDER_DEPLOY_HOOK_URL` is set.

## Deployment

- **Render (free, 512 MB RAM):** `render.yaml` builds the light image with hash embeddings, because CLIP plus torch does not fit in 512 MB.
- **Hugging Face Spaces (free CPU, 16 GB RAM):** build with `WITH_CLIP=true` for CLIP embeddings in production.

## Repository layout

```
backend/app/{api/v1,core,db,schemas,services}   FastAPI app
frontend/{index.html,app.js}                    Tailwind + vanilla JS client (served at /)
data/                                           seed catalog, SQL schema, init script, sample photos
scripts/                                        seed/sample generators, benchmark harness
tests/                                          pytest suite
```
