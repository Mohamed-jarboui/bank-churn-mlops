#!/bin/bash
set -e

if [ "$APP_TYPE" == "ui" ]; then
    echo "🚀 Starting Streamlit UI..."
    exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
else
    echo "🚀 Starting FastAPI Backend..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
