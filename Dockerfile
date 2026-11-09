# syntax=docker/dockerfile:1.4
FROM ghcr.io/vantage6/infrastructure/algorithm-base:4.15

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN --mount=from=ghcr.io/astral-sh/uv:0.8,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-hashes --no-emit-project -o /tmp/requirements.txt && \
    uv pip install --system \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /app

ARG PKG_NAME="src_v6"

# Install your own package without re-resolving/re-downloading dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /app

ENV PKG_NAME=${PKG_NAME}
# ENV TF_USE_LEGACY_KERAS=1
# ENV TF_DISABLE_MLIR_CRASH_REPRODUCER=1
# ENV CUDA_MODULE_LOADING=LAZY


CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"