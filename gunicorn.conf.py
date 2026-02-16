import os


def _get_env(key: str, default: str) -> str:
    value = os.getenv(key)
    if value is not None:
        return value
    return default


bind = _get_env("GUNICORN_BIND", "127.0.0.1:8080")
workers = int(_get_env("GUNICORN_WORKERS", "24"))
threads = int(_get_env("GUNICORN_THREADS", "4"))
timeout = int(_get_env("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(_get_env("GUNICORN_GRACEFUL_TIMEOUT", "120"))
keepalive = int(_get_env("GUNICORN_KEEPALIVE", "5"))
worker_class = _get_env("GUNICORN_WORKER_CLASS", "gthread")
accesslog = _get_env("GUNICORN_ACCESSLOG", "-")
errorlog = _get_env("GUNICORN_ERRORLOG", "-")
loglevel = _get_env("GUNICORN_LOGLEVEL", "info")

# Optional hardening and stability knobs
max_requests = int(_get_env("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(_get_env("GUNICORN_MAX_REQUESTS_JITTER", "100"))

# Because indexes/models loading is heavy, we set True here to fork the workers
# from the master.
preload_app = _get_env("GUNICORN_PRELOAD_APP", "true").lower() in ("1", "true", "yes")
