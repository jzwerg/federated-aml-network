"""Per-node synthetic transaction data.

Each bank generates and persists its OWN small dataset inside its OWN mounted
volume (``/data``). No bank ever reads another bank's data — that isolation is
the core claim of the project. The data here is intentionally trivial; the
fraud-typology generator and the realistic non-IID split are later milestones
(see PLAN.md).
"""

import hashlib
import os
from typing import Tuple

import numpy as np
import torch

from common.model import N_FEATURES

DATASET_FILE = "transactions.npz"
DEFAULT_N_SAMPLES = 512


def _seed_from_bank(bank_id: str) -> int:
    """Deterministic per-bank seed (so each node gets a distinct distribution).

    ``hash()`` is salted per process, so we derive a stable seed from a digest.
    """
    digest = hashlib.sha256(bank_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31)


def prepare_dataset(
    data_dir: str, bank_id: str, n_samples: int = DEFAULT_N_SAMPLES
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (once) and load this bank's synthetic dataset from its own volume."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, DATASET_FILE)

    if not os.path.exists(path):
        rng = np.random.default_rng(_seed_from_bank(bank_id))
        X = rng.normal(size=(n_samples, N_FEATURES)).astype("float32")
        # A simple linear rule + noise produces the (im)balanced labels.
        weights = rng.normal(size=(N_FEATURES,))
        logits = X @ weights + rng.normal(scale=0.5, size=n_samples)
        y = (logits > 0.0).astype("int64")
        np.savez(path, X=X, y=y)

    data = np.load(path)
    return data["X"], data["y"]


def load_tensors(data_dir: str, bank_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return this bank's data as torch tensors ready for training."""
    X, y = prepare_dataset(data_dir, bank_id)
    return torch.from_numpy(X), torch.from_numpy(y)
