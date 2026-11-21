# govscape
Searching millions of .gov PDFs

## Build

To build the govscape server, we use the poetry build system. If poetry is properly installed, then running the following command should build the package:

```bash
poetry install
```

## Run

To run the initial version, you first build the embeddings, indices, etc. with:
```bash
poetry run python scripts/run_embedding_pipeline.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```

Then, you run the RESTful API server with Gunicorn (for production, default worker_class=sync):
```bash
GUNICORN_WORKERS=2 \
poetry run gunicorn -c gunicorn.conf.py 'scripts.serving.start_api_server:create_app()'
```

Or use the wrapper to pass your usual CLI app arguments along with Gunicorn:
```bash
poetry run -- python -m scripts.serving.run_gunicorn \
  -p data/test_data/TechnicalReport234PDFs \
  -d data/test_data \
  -tm ST -vm CLIP -k 20 -i Memory -- \
  'gunicorn -c gunicorn.conf.py scripts.serving.start_api_server:create_app()'
```

Tuning knobs (Gunicorn env vars supported by `gunicorn.conf.py`):
- `GUNICORN_WORKERS`
- `GUNICORN_THREADS` (only applies to gthread workers)
- `GUNICORN_WORKER_CLASS`
- `GUNICORN_TIMEOUT`
- `GUNICORN_MAX_REQUESTS`
- `GUNICORN_MAX_REQUESTS_JITTER`
- `GUNICORN_PRELOAD_APP`

For development, you can still use the simple runner:
```bash
poetry run python scripts/start_api_server.py -p "data/test_data/TechnicalReport234PDFs" -d "data/test_data"
```

### API Documentation

The project includes a RESTful API server built with Flask and documented with Swagger/OpenAPI. To access the API playground:
1. Start the server using the instructions above
2. Visit http://localhost:8080/docs
3. Use the Swagger UI to try out the endpoints
4. Check the response codes and data formats

#### Adding New Endpoints

When adding new endpoints to the API:

1. Define the request/response models using Flask-RESTX fields
2. Create a new Resource class in the appropriate namespace
3. Use the `@ns.doc()` and `@ns.response()` decorators for documentation
4. Add example requests/responses in the Swagger UI


#### Govscape Source COOP Repository
The data that govscape is derived from lives at the following source.coop [repository](https://source.coop/govscape/eota-pdf-archive).
All derived data (page text and images, embeddings, metadata, and search indices) lives at the following source.coop [repository](https://source.coop/govscape/eota-derivative-data).

Scripts default to reading PDFs from the archive (`--pdf_bucket_name`) and reading/writing derived data in the derivative repository (`--bucket_name`). Both are accessed through the source.coop S3 proxy (`https://data.source.coop`, bucket `govscape`); pass `--endpoint_url ''` to use AWS S3 instead. Both repositories are publicly readable. Requests to the proxy are signed only with explicitly selected credentials (`--profile`/`--pdf_profile`, `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`), otherwise they are sent anonymously; ambient AWS credentials such as an EC2 instance role or the default profile are ignored, since the proxy rejects them. Writing requires source.coop-issued credentials, stored as a profile in `~/.aws/credentials`:

```
[eota-pdf-derivative-data]
aws_access_key_id = ...
aws_secret_access_key = ...
aws_session_token = ...
endpoint_url = https://data.source.coop
```

Select the profile with `AWS_PROFILE=<name>` or per bucket with `--profile` (derived data) and `--pdf_profile` (PDF archive).
