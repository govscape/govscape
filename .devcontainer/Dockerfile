# The dev container is based on the same base image as the CI environment and the main image for the API server.

FROM ghcr.io/bcglee/govscape-base:py3.11.14-lucene10.0.0-poetry2.3.2

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

# Add dev tools here.
# Node, git, aws and a few other tools are mounted using devcontainer.json features.
RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
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
