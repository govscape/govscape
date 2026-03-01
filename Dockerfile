# GovScape image for both API and embedding servers.
# Use ./compose.yaml to start the API server after running the embedding pipeline.
FROM ghcr.io/bcglee/govscape-base:sha-2c5d3a9

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/govscape

COPY . .

# Install poetry and dependencies. --no-cache still produces /root/.cache/artifacts/
RUN curl -fsSL https://install.python-poetry.org | python3 - \
    && poetry install --no-cache \
    && rm -rf /root/.cache/*

EXPOSE 8080
