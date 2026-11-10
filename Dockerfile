FROM node:24-bookworm-slim AS web
WORKDIR /build
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/package.json
RUN npm ci
COPY apps/web ./apps/web
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.6 \
    && useradd --uid 10001 --create-home studio
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY data/personas ./data/personas
COPY --from=web /build/apps/web/dist ./apps/web/dist
ENV BIAS_CORPUS_DIR=/app/data/personas/elon-musk \
    BIAS_WEB_DIR=/app/apps/web/dist \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER studio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["/app/.venv/bin/bias", "serve", "--host", "0.0.0.0", "--port", "8000"]

