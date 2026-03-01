# The dev container is based on the same base image as the CI environment and the main image for the API server.

# TODO, once first PR is merged, update to use proper version tag instead of sha for the base image
FROM ghcr.io/bcglee/govscape-base:sha-2c5d3a9

# In the docker environment, we can install dependencies globally without conflicts.
# This allows us to use a globally installed pylucene package, which cannot be pip installed.
ENV POETRY_VIRTUALENVS_CREATE=false

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

# In the docker environment, we can install dependencies globally without conflicts.
# This allows us to use a globally installed pylucene package, which cannot be pip installed.
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_HOME=/opt/poetry
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -
