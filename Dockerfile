#-----------------------------------------------------
# Builder (dependencies via uv)
#-----------------------------------------------------

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

#-----------------------------------------------------
# Runtime
#-----------------------------------------------------

FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY main.py index.py config.py ./
COPY components ./components
COPY models ./models
COPY pipelines ./pipelines
COPY services ./services

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python3", "main.py"]