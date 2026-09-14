# Changelog

## 1.1.0 (2026-09-13)

### Added
- **Alternative plans.** Every `/spec` and `/bom` response includes up to `MAX_ALTERNATIVES` (default 3) plans in the next-best finishes, solved under the same constraints and compliance checks. The UI shows them as cards with the price difference and a "Use this finish" button.
- **Layout priority.** New `layout_priority` constraint: `fill_width` (default, previous behaviour) or `complete_kitchen`, which prefers a shorter run whose upper cabinets cover it when the budget can't do both.
- **Saved, shareable specifications.** Results are stored in a new `spec_runs` table (`SAVE_SPECS`). New endpoints `GET /api/v1/specs` and `GET /api/v1/specs/{request_id}`. The UI puts `?spec=<id>` in the address bar, restores results and form from share links, and lists recent specs.
- **Printable quote / PDF.** Print stylesheet with a quote header (id, date, constraints, share link) and a "Print / PDF" button.
- **Security options.** `API_KEYS` (X-API-Key on `/spec`, `/bom`, `/search`), `RATE_LIMIT_PER_MINUTE` (sliding window, 429 + `Retry-After`), `CORS_ORIGINS` allow-list. The UI prompts for a key when the server requires one.
- **Catalog admin API** behind `ADMIN_API_KEYS`: `PATCH /api/v1/admin/catalog/{part_id}` (price, stock), `PUT /api/v1/admin/catalog` (validated bulk upsert with re-embedding), `POST /api/v1/admin/reindex`.
- **Prometheus metrics** at `/metrics` (`METRICS_ENABLED`): request counts and latency by route template, per-stage pipeline latency, spec status counts, LLM outcome counts.
- `finish_style_score` in responses.
- 17 new tests (`tests/test_features.py`); 80 in total.

### Changed
- `request_id` is now 16 hex characters (was 12), since it doubles as a share-link id.
- Hybrid retriever split into vector search plus per-finish SQL assembly (`cabinet_modules_for_finish`), so alternatives reuse one vector search.
- CORS now allows only the methods and headers the API uses.

### Fixed
- Qdrant collections are namespaced by embedding backend (`cabinet_modules_clip`, `cabinet_modules_hash`); both backends are 512-d, so a shared collection would have silently mixed incompatible vectors.

## 1.0.0 (2026-09-13)

Initial handoff: OpenCV preprocessing, CLIP + sparse hybrid retrieval with RRF, SQL hard constraints, knapsack layout solver, grounded BOM generation, web UI, Docker Compose, CI, benchmarks.
