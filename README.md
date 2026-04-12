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
poetry run -- python -m scripts/serving/run_gunicorn.py \
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
