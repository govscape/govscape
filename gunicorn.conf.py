import multiprocessing
import os

def _get_env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default

bind = _get_env("GUNICORN_BIND", "0.0.0.0:8080")
workers = int(_get_env("GUNICORN_WORKERS", str(max(2, multiprocessing.cpu_count() * 2 + 1))))
threads = int(_get_env("GUNICORN_THREADS", "4"))
timeout = int(_get_env("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(_get_env("GUNICORN_GRACEFUL_TIMEOUT", "120"))
keepalive = int(_get_env("GUNICORN_KEEPALIVE", "5"))
worker_class = _get_env("GUNICORN_WORKER_CLASS", "sync")
accesslog = _get_env("GUNICORN_ACCESSLOG", "-")
errorlog = _get_env("GUNICORN_ERRORLOG", "-")
loglevel = _get_env("GUNICORN_LOGLEVEL", "info")

# Optional hardening and stability knobs
max_requests = int(_get_env("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(_get_env("GUNICORN_MAX_REQUESTS_JITTER", "100"))

# Preload app: False by default because models are heavy; set True if we want faster worker boot
preload_app = _get_env("GUNICORN_PRELOAD_APP", "false").lower() in ("1", "true", "yes")
