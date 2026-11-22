"""Small helpers: seeding, device selection and a CSV + TensorBoard logger."""
from __future__ import annotations

import csv
import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Logger:
    """Writes one row per epoch to `progress.csv` and mirrors it to TensorBoard."""

    def __init__(self, out_dir: str, use_tensorboard: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "progress.csv")
        self._rows: list = []
        self._fields: list = []
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(os.path.join(out_dir, "tb"))
            except Exception as e:  # tensorboard not installed -> CSV only
                print(f"[logger] TensorBoard disabled ({e})")

    def log(self, row: Dict[str, float], step: int) -> None:
        # columns can appear later (e.g. model stats after the first model fit),
        # so keep all rows and rewrite the (small) file each time
        self._rows.append(dict(row))
        self._fields += [k for k in row if k not in self._fields]
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            w.writerows(self._rows)
        if self._tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self._tb.add_scalar(k, v, step)
            self._tb.flush()

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
