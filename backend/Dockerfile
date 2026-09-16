# Build from the repository root:  docker build -f backend/Dockerfile -t roomspec-ai .
# CLIP image (~2 GB, needs >1 GB RAM):   --build-arg WITH_CLIP=true
FROM python:3.12-slim AS base

ARG WITH_CLIP=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    PORT=8000

# libcairo2 is required by cairosvg (used by the FIT feature to rasterize
# catalog SVG thumbnails before sending them to an image-editing API).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libcairo2 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements-ml.txt backend/
RUN if [ "$WITH_CLIP" = "true" ]; then \
      pip install -r backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu \
      && python -c "from transformers import CLIPModel, CLIPProcessor as P; m='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(m); P.from_pretrained(m)"; \
    else \
      pip install -r backend/requirements.txt; \
    fi

COPY backend/app backend/app
COPY frontend frontend
COPY data/seed_modules.json data/catalog_modules.sql data/init_db.py data/
COPY data/samples data/samples

RUN useradd --create-home --uid 10001 roomspec \
 && chown -R roomspec /app \
 && mkdir -p /opt/hf-cache && chown -R roomspec /opt/hf-cache
USER roomspec

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/api/v1/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT} --proxy-headers"]
