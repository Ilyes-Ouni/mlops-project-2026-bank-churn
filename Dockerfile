# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serve.txt \
    && pip install --no-cache-dir --no-deps .

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser pyproject.toml README.md params.yaml ./

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "bank_churn.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
