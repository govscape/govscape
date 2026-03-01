# GovScape image for both API and embedding servers.
# Use ./compose.yaml to start the API server after running the embedding pipeline.

# TODO, once first PR is merged, update to use proper version tag instead of sha for the base image
FROM ghcr.io/bcglee/govscape-base:sha-2c5d3a9

# In the docker environment, we can install dependencies globally without conflicts.
# This allows us to use a globally installed pylucene package, which cannot be pip installed.
ENV POETRY_VIRTUALENVS_CREATE=false

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/govscape

COPY . .

# Install poetry and dependencies. --no-cache still produces /root/.cache/artifacts/
RUN curl -fsSL https://install.python-poetry.org | python3 - \
    && poetry install --no-cache \
    && rm -rf /root/.cache/*

EXPOSE 8080
