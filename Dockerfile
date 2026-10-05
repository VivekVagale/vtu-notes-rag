# Service image for the VTU Notes RAG API.
#
# NOT BUILT ON THE AUTHORING MACHINE - Docker is not installed there. The
# dependency set in requirements-service.txt was verified in a clean virtual
# environment, but this Dockerfile itself is untested. Expect to adjust it.
#
#   docker build -t vtu-notes-rag .
#   docker run -p 8000:8000 \
#     -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=... \
#     -e OWNER_EMAIL=you@example.com -e ALLOW_DEV_LOGIN=false \
#     -e CORS_ORIGINS=https://your-site.example \
#     -v vtu-data:/app/data \
#     vtu-notes-rag

FROM python:3.12-slim

# PyMuPDF wheels are self-contained; curl is only here for the health check.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# The index, the registry and uploaded files all live here. This MUST be a
# persistent volume - without it every restart loses the library.
VOLUME ["/app/data"]

ENV EMBED_BACKEND=onnx \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf

# Bake the embedding model into the image so the first request is not the slow
# one and the container does not need the network at boot.
RUN python -c "from huggingface_hub import hf_hub_download as d; \
    d('BAAI/bge-small-en-v1.5','onnx/model.onnx'); \
    d('BAAI/bge-small-en-v1.5','tokenizer.json')"

EXPOSE 8000

# Cold start is ~3s; give the health check room for a slow disk.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# One worker on purpose: the ingest worker and the vector index want a single
# writer. Scale by putting more instances behind a load balancer only after
# moving the index off local disk.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
