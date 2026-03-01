# GovScape image for both API and embedding servers.
# Use ./compose.yaml to start the API server after running the embedding pipeline.

# TODO, once first PR is merged, update to remove the branch from the version tag
FROM ghcr.io/bcglee/govscape-base:py3.11.14-lucene10.0.0-poetry2.3.2-lucene-keyword-index

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/govscape

COPY . .

# Install poetry and dependencies. --no-cache still produces /root/.cache/pypoetry/artifacts/
RUN poetry install --no-cache \
    && rm -rf /root/.cache/pypoetry/artifacts/

EXPOSE 8080
