# The dev container is based on the same base image as the CI environment and the main image for the API server.

# TODO, once first PR is merged, update to remove the branch from the version tag
FROM ghcr.io/bcglee/govscape-base:py3.11.14-lucene10.0.0-poetry2.3.2-lucene-keyword-index

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

# Setup any dev tools here.
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
    && rm -rf /var/lib/apt/lists/*
