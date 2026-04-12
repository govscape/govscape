# Govscape: Contributing Guide

Thank you for your interest in contributing! Please read the following guidelines to help us maintain a high-quality, collaborative codebase.

## Code of Conduct

We adhere to the [Python Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).

## Collaboration Practices

For those who are new to the process of contributing code, welcome! We value your contribution, and are excited to work with you. GitHub's [pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) will walk you through how to file a PR.

Please follow the [SciML Collaborative Practices](https://docs.sciml.ai/ColPrac/stable/) and [GitHub Collaborative Practices](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes) guides to help make your PR easier to review.

In this repo, please use the convention <initials>/<branch-name> for pull request branch names, e.g. ms/scheduler-pass.
This way in bash when you type your initials git checkout ms/ and <tab> you can see all your branches. We will use other names for special purposes.

### Packaging

Govscape uses [poetry](https://python-poetry.org/) for packaging and dependency management.

To install for development, clone the repository and run:
```bash
poetry install
```
to install the current project and dev dependencies.

### Dev container
We recommend using the dev container for local development to ensure your local environment has all necessary dependencies set up correctly.
The dev container is required for using the Lucene keyword index, and is also the easiest way to get up and running without needing to manually install and manage dependencies on your machine.

You will need [Docker Desktop](https://www.docker.com/get-started/) and the [Dev Containers extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
Dev containers also work with the [Remote - SSH extension](https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host) if you use a remote host machine.

Clone the repository (ideally with https and not ssh) and open it in VS Code.
You should see a prompt on the bottom right of the screen to "Reopen in Container".
When clicked, VS Code relaunches and builds the dev container; this will take a few minutes the first time. Once the window reloads, you will be connected to the dev container and can run all commands from the terminal within VS Code. `poetry install` automatically starts when the dev container build finishes.

Poetry will not use the workspace .venv folder, but will create an isolated virtual environment within the dev container with lucene and all other dependencies installed.
This will not impact any Python environments on your host machine.

AWS credentials at `~/.aws` on your host machine are automatically mounted onto the dev container. Both `aws` and `poetry run s5cmd` commands work without any additional configuration.

`docker` commands within the dev container interact with the Docker daemon on your host machine.
Note that if you need to bind a directory in the workspace, the path must be from the host, not from within the dev container.
You can use `${LOCAL_WORKSPACE_FOLDER}` to refer to the workspace folder on the host machine from within the dev container.
Alternatively, use an external terminal to run `docker` commands on the host machine if needed.

If you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host machine, you will have GPUs available within the dev container.
You can verify this by running `nvidia-smi` within the dev container.

Node.js is included in the dev container. You can run `npm` commands within the dev container terminal.

The dev container automatically configures Git, copying your `~/.gitconfig` file. You can use Git CLI commands within the dev container terminal, or use the Git integration in VS Code.
If you use Git over ssh, you may need to [enable SSH agent forwarding](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials) to push from within the dev container.

VS Code forwards ports 8080 and 5173 for the API and web servers respectively, so you can access them from your local browser if they are running within the dev container.
Note that the frontend by default expects the API server to exist at `localhost:8080`.

If you ever need to exit the dev container and return VS Code to your local environment, open the command palette in VS Code (Ctrl+Shift+P or Cmd+Shift+P) and select "Dev Containers: Reopen Folder Locally".
If you use WSL, you can also select "Dev Containers: Reopen in WSL" to open the folder in WSL instead of Windows.
"Dev Containers: Reopen in Container" will bring you back into the dev container.

### Pre-commit hooks

Pull requests must pass some formatting, linting, and typing checks before we can merge them. These checks can be run automatically before you make commits, which is why they are sometimes called "pre-commit hooks". We use [pre-commit](https://pre-commit.com/) to run these checks. In the dev container, pre-commit is already installed.

To install pre-commit hooks to run before committing, run:
```bash
poetry run pre-commit install
```
If you prefer to instead run pre-commit hooks manually, run:
```bash
poetry run pre-commit run -a
```

### Testing
Govscape uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
poetry run pytest
```

- Tests are located in the `govscape/tests/` directory at the project root.
- Write thorough tests for your new features and bug fixes.

#### Optional Static Type Checking

The pytest will run mypy to check for type errors, so you shouldn't need to run it manually.
In case you do need to run mypy manually, you can do so with:

```bash
poetry run mypy .
```

## Data Model

Because the data that lives on the remote backend (i.e. AWS S3) is a core aspect of this project, we need clear documentation for how it is structured. This documentation lives in `DATA_MODEL.md`, and you should take a look at it to familiarize yourself with it. Any changes that alter this structure should update the data model file to make sure that it remains up to date.

## Code Style
### Assertions and Validation

- **Do not use `assert` statements for user-facing validation.**
    - `assert` statements are removed when Python is run with the `-O` (optimize) flag.
    - Use explicit error handling (e.g., `if ...: raise ValueError(...)`) for all user-facing functions, following the [array API specification](https://data-apis.org/array-api/latest/).
    - user-facing functions are anything exposed from `__all__` in the toplevel `__init__.py`
- `assert` statements may be used for internal debugging, invariants, and sanity checks that are not critical to production behavior.

### Getters and Setters
- Use `@property` decorators for getters and setters.
- This means you may need to define a private `_foo` attribute in your dataclass to implement the `foo` property.
- Avoid using `get_` and `set_` prefixes in method names.
- `get_` and `set_` prefixes are allowed for global getters and setters, such as `util.get_version()`.

### Logging

Use the standard library `logging` module rather than `print` for all diagnostic output:

```python
import logging

logging.info("Stage started")
logging.warning("Something unexpected happened")
logging.error("Something failed: %s", err)
```

`print` bypasses the logging configuration and cannot be silenced, filtered by level, or redirected by log handlers. The root logger is configured in `pdf_processing_pipeline.py` with `INFO` level — use `logging.debug` for verbose output that should be off by default, and `logging.info` for normal progress messages.


---
**If you find an error or unclear section, please fix it or open an issue.**


## Running Govscape Locally

To do this, you need to start by creating a directory within govscape/data that holds a set of PDFs and one that holds a `pdf_metadata.parquet` file. We will assume that these directories are named `govscape/data/s3_mock/archive/PDFs/` and `govscape/data/s3_mock/archive/CDX/pdf_metadata.parquet`.

You can pull this data from the S3 bucket by using:

```
poetry run python scripts/data_prep/download_test_data.py \
    --bucket_name bcgl-public-bucket \
    --local_base_dir data/s3_mock \
    --num_pdfs 500
```


### Creating the embeddings

To create the (dummy) embeddings & additional metadata, first run:

```
poetry run python scripts/pipeline/run_embedding_pipeline.py --num_pages_to_process 5 \
    --batch_size 100 --backend 'local' --local_base_dir 'data/s3_mock' --pdf_dir 'archive/PDFs/' \
    --remote_data_dir "test-serving" --text_model_type 'Dummy' --visual_model_type 'Dummy'
```

Next, create the indices:
```

# Run the embeddings pipeline
poetry run python scripts/indexing/generate_index_embedding.py --num_pages_to_process 10 --backend 'local' --local_base_dir 'data/s3_mock' --embedding_prefix "embeddings" --remote_data_dir 'test-serving' --out_index_prefix 'index'

poetry run python scripts/indexing/generate_index_embedding.py --num_pages_to_process 10 --backend 'local' --local_base_dir 'data/s3_mock' --embedding_prefix "embeddings_img_pg" --remote_data_dir 'test-serving'  --out_index_prefix 'index_img_pg'

poetry run python scripts/indexing/generate_index_keyword.py --num_pages_to_process 10 --backend 'local' --local_base_dir 'data/s3_mock' --remote_data_dir 'test-serving' --keyword_index_type 'SQLite'

poetry run python scripts/indexing/generate_index_metadata.py --num_pages_to_process 10 --backend 'local' --local_base_dir 'data/s3_mock' --remote_data_dir 'test-serving' --cdx_parquet_key  'archive/CDX/pdf_metadata.parquet'
```

At this point, all of the indices required to run the API server have been created. To start the API server locally, run:

```
poetry run python -m scripts.serving.run_gunicorn --backend 'local' --local_base_dir 'data/s3_mock' --remote_data_directory 'test-serving' --text_model 'Dummy' --visual_model 'Dummy' --keyword_index_type 'SQLite' --vector_index_type 'Memory'
```

To start the web server locally, in a separate terminal run:

```
cd interface && npm install && npm run dev --open
```

Within a dev container web server, add `-- --host`:

```
cd interface && npm install && npm run dev -- --host
```
