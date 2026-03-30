import atexit
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .distributed import get_machine_local_and_dist_rank
from .general import safe_makedirs


class WandbLogger:
    """A minimal logger wrapper exposing the same interface as TensorBoardLogger."""

    def __init__(
        self,
        path: str,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        mode: str = "online",
        resume: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._run = None
        self._path = path
        _, self._rank = get_machine_local_and_dist_rank()

        if self._rank != 0:
            logging.debug("Not logging to Weights & Biases because rank %s != 0", self._rank)
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` to use logging.use_wandb=True."
            ) from exc

        safe_makedirs(path)
        logging.info("Weights & Biases run directory: %s", path)
        self._run = wandb.init(
            dir=path,
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            resume=resume,
            config=config,
            **kwargs,
        )
        atexit.register(self.close)

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        if self._run is not None:
            self._run.log({}, commit=False)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        if self._run is None:
            return
        self._run.log(payload, step=step)

    def log(self, name: str, data: Any, step: int) -> None:
        if self._run is None:
            return
        self._run.log({name: data}, step=step)

    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray, Any],
        step: int,
        fps: int = 4,
    ) -> None:
        if self._run is None:
            return

        import wandb

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        if data.ndim == 3:
            self._run.log({name: wandb.Image(data)}, step=step)
        elif data.ndim == 5:
            # Wandb.Video expects (T, C, H, W) or (B, T, C, H, W). Use the first sample if batched.
            video = data[0] if data.shape[0] > 1 else data.squeeze(0)
            self._run.log({name: wandb.Video(video, fps=fps)}, step=step)
        else:
            raise ValueError(
                f"Unsupported data dimensions: {data.ndim}. Expected 3D for images or 5D for videos."
            )
