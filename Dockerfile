# ── base: dependências compartilhadas ────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── api: backend FastAPI ──────────────────────────────────────────────────────
FROM base AS api

COPY api.py .
COPY model_artifacts/ ./model_artifacts/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── ui: frontend Streamlit ────────────────────────────────────────────────────
FROM base AS ui

COPY streamlit_app.py .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
