"""Synthetic transactions, fraud typologies, and the non-IID bank split.

The signal this product is about lives *between* banks: laundering typologies
each light up a different group of features, and a mule ring's hops are scattered
one-per-bank. A model that has only ever seen one bank's slice has blind spots; a
federated model that averages across all three covers them. ``benchmark/`` proves
that empirically.

Isolation note: this is a *simulation*. Each bank deterministically regenerates the
same global population (shared seed) and selects only its own rows — and on disk a
bank's volume only ever contains its own slice. In a real deployment each node
would physically hold only its own data; the shared generator is a convenience so
the typologies line up across nodes. No bank ever reads another's rows.

Trivial-but-real on purpose: realistic time-series dynamics and richer typologies
are later polish (see ``PLAN.md`` / ``ROADMAP.md``); DP and the attack are too.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from common.model import N_FEATURES

DATASET_FILE = "transactions.npz"
GLOBAL_SEED = 20260601  # fixed so every node agrees on the population

# Each typology lights up a disjoint group of features; the last group is
# non-discriminative noise shared by all rows. A solo model only learns the
# groups whose positives it has actually seen.
FEATURE_GROUPS: Dict[str, List[int]] = {
    "structuring": [0, 1, 2],
    "layering": [3, 4, 5],
    "mule_ring": [6, 7, 8],
    # 9, 10, 11 -> background noise (amount, time-of-day, etc.)
}
FRAUD_TYPOLOGIES = list(FEATURE_GROUPS.keys())
SIGNAL = 1.6  # mean shift applied to a typology's feature group

# Population sizes (kept small — first real data, not a benchmark suite).
N_LEGIT = 1500
N_PER_TYPOLOGY = 500
TEST_FRACTION = 0.2

# How each typology's *training* rows are distributed across banks (non-IID).
# Structuring concentrates in bank-a, layering in bank-b; the mule ring is split
# evenly — its hops are cross-institutional, so no single bank sees enough of it.
BANK_IDS = ["bank-a", "bank-b", "bank-c"]
TYPOLOGY_BANK_WEIGHTS: Dict[str, List[float]] = {
    "structuring": [0.70, 0.15, 0.15],
    "layering": [0.15, 0.70, 0.15],
    "mule_ring": [1 / 3, 1 / 3, 1 / 3],
}
LEGIT_BANK_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]


def _make_rows(rng: np.random.Generator, n: int, typology: str | None) -> np.ndarray:
    """n feature rows for one class (``None`` = legit baseline)."""
    X = rng.normal(size=(n, N_FEATURES)).astype("float32")
    if typology is not None:
        for idx in FEATURE_GROUPS[typology]:
            X[:, idx] += SIGNAL
    return X


def generate_population() -> Dict[str, np.ndarray]:
    """Build the full, deterministic transaction population.

    Returns arrays ``X`` (features), ``y`` (1 = fraud), and ``typ`` (string label
    per row: 'legit' or a typology name).
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    blocks_X = [_make_rows(rng, N_LEGIT, None)]
    blocks_y = [np.zeros(N_LEGIT, dtype="int64")]
    blocks_t = [np.array(["legit"] * N_LEGIT)]

    for typ in FRAUD_TYPOLOGIES:
        blocks_X.append(_make_rows(rng, N_PER_TYPOLOGY, typ))
        blocks_y.append(np.ones(N_PER_TYPOLOGY, dtype="int64"))
        blocks_t.append(np.array([typ] * N_PER_TYPOLOGY))

    X = np.concatenate(blocks_X)
    y = np.concatenate(blocks_y)
    typ = np.concatenate(blocks_t)

    # Deterministic shuffle so the test split is a representative mix.
    perm = np.random.default_rng(GLOBAL_SEED + 1).permutation(len(y))
    return {"X": X[perm], "y": y[perm], "typ": typ[perm]}


def _stratified_test_mask(typ: np.ndarray) -> np.ndarray:
    """Hold out TEST_FRACTION of every class for a balanced global test set."""
    rng = np.random.default_rng(GLOBAL_SEED + 2)
    mask = np.zeros(len(typ), dtype=bool)
    for label in np.unique(typ):
        idx = np.where(typ == label)[0]
        n_test = int(round(len(idx) * TEST_FRACTION))
        mask[rng.choice(idx, size=n_test, replace=False)] = True
    return mask


def _assign_banks(typ_train: np.ndarray) -> np.ndarray:
    """Assign each training row to a bank per the non-IID weights above."""
    rng = np.random.default_rng(GLOBAL_SEED + 3)
    banks = np.empty(len(typ_train), dtype=object)
    for label in np.unique(typ_train):
        idx = np.where(typ_train == label)[0]
        weights = (
            LEGIT_BANK_WEIGHTS if label == "legit" else TYPOLOGY_BANK_WEIGHTS[label]
        )
        banks[idx] = rng.choice(BANK_IDS, size=len(idx), p=weights)
    return banks


def split_dataset() -> Dict[str, object]:
    """Partition the population into per-bank train slices + a shared test set.

    Used by the benchmark (which legitimately needs the global view) and by each
    client to extract *only* its own slice.
    """
    pop = generate_population()
    test_mask = _stratified_test_mask(pop["typ"])

    train = {k: v[~test_mask] for k, v in pop.items()}
    test = {k: v[test_mask] for k, v in pop.items()}

    bank_of = _assign_banks(train["typ"])
    banks: Dict[str, Dict[str, np.ndarray]] = {}
    for bank_id in BANK_IDS:
        sel = bank_of == bank_id
        banks[bank_id] = {
            "X": train["X"][sel],
            "y": train["y"][sel],
            "typ": train["typ"][sel],
        }
    return {"banks": banks, "test": test}


def prepare_dataset(
    data_dir: str, bank_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize (once) and load THIS bank's own slice from its own volume."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, DATASET_FILE)

    if not os.path.exists(path):
        slice_ = split_dataset()["banks"][bank_id]
        # Only this bank's rows are ever written to this bank's volume.
        np.savez(path, X=slice_["X"], y=slice_["y"], typ=slice_["typ"])

    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def load_tensors(data_dir: str, bank_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """This bank's data as torch tensors ready for training."""
    X, y = prepare_dataset(data_dir, bank_id)
    return torch.from_numpy(X), torch.from_numpy(y)
