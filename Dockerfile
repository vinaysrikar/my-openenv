# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Hugging Face Spaces run as user 1000
RUN useradd -m -u 1000 hfuser && chown -R hfuser:hfuser /app
USER 1000

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server on the port HF Spaces expects
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
