import logging
import threading
from typing import Any

import torch


class LazyDeviceModel:
    """Mixin for torch-backed embedding models that load ``self.model`` on the CPU
    and choose the device (moving the model onto it) on first use.

    Gunicorn loads the server in its master process and forks the workers from
    it (``preload_app``). Initialising CUDA in the master, which even
    ``torch.cuda.is_available()`` does, leaves CUDA unusable in the forked
    workers. Deferring the device choice keeps the master CUDA-free, so each
    worker initialises CUDA itself.

    Every worker holds its own copy of the model on the GPU, so a GPU can run out
    of memory once enough workers have moved theirs. A worker whose model doesn't
    fit falls back to the CPU instead of failing its requests.
    """

    model: Any
    _device: torch.device | None = None
    _device_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        if self._device is None:
            with self._device_lock:
                if self._device is None:
                    self._device = self._place_model()
        return self._device

    def _place_model(self) -> torch.device:
        cpu = torch.device("cpu")
        if not torch.cuda.is_available():
            self._move_model_to(cpu)
            return cpu
        try:
            self._move_model_to(torch.device("cuda"))
            return torch.device("cuda")
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                "Not enough GPU memory for %s; running it on the CPU instead. "
                "Reduce GUNICORN_WORKERS to keep every worker on the GPU.",
                type(self).__name__,
                exc_info=True,
            )
        # The failed move may have left the model split across devices and, for
        # models that run in half precision on the GPU, already converted. Move it
        # off the GPU before converting back to float32 (what the models are
        # loaded in, and what runs fast on the CPU): converting on the GPU would
        # need more GPU memory.
        self.model.to(cpu)
        self.model.float()
        torch.cuda.empty_cache()
        return cpu

    def _move_model_to(self, device: torch.device) -> None:
        self.model.to(device)
