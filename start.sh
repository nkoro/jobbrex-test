#!/bin/bash
# EvalOps Lab — Replit unified launcher
# Runs FastAPI backend (port 8000, internal) + Streamlit frontend (port $PORT, external)

set -e

echo "==> Installing backend deps..."
pip install -r requirements.txt -q --no-warn-script-location

echo "==> Installing frontend deps..."
pip install -r frontend/requirements.txt -q --no-warn-script-location

echo "==> Starting FastAPI backend on port 8000..."
uvicorn backend_api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Give uvicorn a few seconds to start before streamlit tries to call it
sleep 4

echo "==> Starting Streamlit frontend on port ${PORT:-8501}..."
cd frontend
BACKEND_URL=http://localhost:8000 \
  streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.headless true \
    --server.address 0.0.0.0

# If streamlit exits, kill uvicorn too
kill $BACKEND_PID 2>/dev/null || true
