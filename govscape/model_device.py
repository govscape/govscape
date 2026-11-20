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
    """

    model: Any
    _device: torch.device | None = None
    _device_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        if self._device is None:
            with self._device_lock:
                if self._device is None:
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    self._move_model_to(device)
                    self._device = device
        return self._device

    def _move_model_to(self, device: torch.device) -> None:
        self.model.to(device)
