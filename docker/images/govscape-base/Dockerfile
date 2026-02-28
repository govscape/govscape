# govscape base image for both API and embedding servers.

# The GitHub action has a build matrix that specifies the python base tag.
# This is the fallback.
ARG PYTHON_VERSION=3.11.14-trixie

# We use a separate build phase to reduce the size of the final image.
FROM python:${PYTHON_VERSION} AS builder

ARG LUCENE_VERSION=10.0.0
ARG LUCENE_SHA256="100c3d61d6799ac16b7b8c1826cddf07fb1715141ebdb0d7b8119cdd96b24574  pylucene-10.0.0-src.tar.gz"
ARG POETRY_VERSION=2.1.2

ENV DEBIAN_FRONTEND=noninteractive

# Install poetry, and dependencies for the pylucene build.
RUN pip install --upgrade pip setuptools wheel build \
    && pip install poetry==${POETRY_VERSION} \
    && apt-get update \
    && apt-get install -y \
        build-essential \
        default-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/jvm/default-java /usr/lib/jvm/temurin

# Download pylucene, verify, build, and install.
WORKDIR /usr/src/pylucene
RUN curl -fsSL https://downloads.apache.org/lucene/pylucene/pylucene-${LUCENE_VERSION}-src.tar.gz -o pylucene-${LUCENE_VERSION}-src.tar.gz \
    && echo "${LUCENE_SHA256}" | sha256sum -c - \
    && tar -xz --strip-components=1 -f pylucene-${LUCENE_VERSION}-src.tar.gz \
    && cd jcc \
    && NO_SHARED=1 JCC_JDK=/usr/lib/jvm/temurin python -m build -nw \
    && pip install dist/*.whl \
    && cd .. \
    && make all install JCC='python -m jcc' PYTHON=python NUM_FILES=16 MODERN_PACKAGING=true

# New stage for runtime. Only preserve necessary files.
# Python, poetry, and pylucene remain after copying.
FROM python:${PYTHON_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/jvm/default-java /usr/lib/jvm/temurin

COPY --from=builder /usr/local /usr/local
