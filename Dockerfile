FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    RECSYS_SERVING_DIR=/app/data/processed/serving

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
# Serving artifacts are produced by `make serve` and are not in git (H&M data rules)
COPY data/processed/serving ./data/processed/serving

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "recsys.api:app", "--host", "0.0.0.0", "--port", "8000"]
