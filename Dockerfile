# govscape image for both API and embedding servers.
# Use compose.yaml to start the API server after running the embedding pipeline.
FROM ghcr.io/bcglee/govscape-base:sha-dcf81b6

# In the docker environment, we can install dependencies globally without conflicts.
# This allows us to use a globally installed pylucene package, which cannot be pip installed.
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /home/govscape

COPY . .

# Install dependencies, then clear the cache to reduce image size.
RUN poetry install && poetry cache clear --all

EXPOSE 8080
