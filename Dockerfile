# govscape image for both API and embedding servers.
# Use compose.yaml to start the API server after running the embedding pipeline.

# govscape currently requires python 3.11.14
ARG PYTHON_VERSION=3.11.14-trixie

# We use a separate build phase to reduce the size of the final image.
# The builder builds Pylucene, and installs necessary python packages.
# Then, the necessary files are copied back to a fresh base image.
FROM python:${PYTHON_VERSION} AS builder

ARG LUCENE_VERSION=10.0.0
ARG POETRY_VERSION=2.1.2

# In the docker environment, we can install dependencies globally without conflicts.
# This allows us to use a globally installed pylucene package, which cannot be pip installed.
ENV POETRY_VIRTUALENVS_CREATE=false

# Install poetry, and dependencies for the pylucene build.
RUN pip install --upgrade pip setuptools wheel \
    && pip install poetry==${POETRY_VERSION} \
    && apt-get update && apt-get install -y \
        build-essential \
        default-jdk

WORKDIR /usr/lib/jvm

# Symlink the java installation so we don't need to modify pylucene scripts.
RUN ln -s default-java temurin

# TODO I think I can get away with removing the second setuptools install, but that may depend on
# if the pip upgrade is necessary first.
RUN pip install build setuptools

WORKDIR /usr/src/pylucene

# TODO: verify checksum before extraction.
RUN curl https://downloads.apache.org/lucene/pylucene/pylucene-${LUCENE_VERSION}-src.tar.gz \
    | tar -xz --strip-components=1

# Build pylucene
RUN cd jcc \
    && NO_SHARED=1 JCC_JDK=/usr/lib/jvm/temurin python -m build -nw \
    && pip install dist/*.whl

RUN make all install JCC='python -m jcc' PYTHON=python NUM_FILES=16 MODERN_PACKAGING=true

# Finally, govscape setup.
RUN mkdir /home/govscape

WORKDIR /home/govscape

COPY . .
RUN poetry install

# New stage for runtime. Only preserved layers are below.
FROM python:${PYTHON_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VIRTUALENVS_CREATE=false

# JRE for lucene.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/lib/jvm
RUN ln -s default-java temurin

# Copy over govscape files, and binaries from the builder stage.
WORKDIR /home/govscape

COPY --from=builder \
    /usr/local \
    /usr/local

COPY --from=builder \
    /home/govscape \
    /home/govscape

EXPOSE 8080
