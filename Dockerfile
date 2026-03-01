# GovScape image for both API and embedding servers.
# Use ./compose.yaml to start the API server after running the embedding pipeline.

# TODO, once first PR is merged, update to use proper version tag instead of sha for the base image
FROM ghcr.io/bcglee/govscape-base:sha-30e2697

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/govscape

COPY . .

# Install poetry and dependencies. --no-cache still produces /root/.cache/artifacts/
RUN poetry install --no-cache \
    && rm -rf /root/.cache/*

EXPOSE 8080
