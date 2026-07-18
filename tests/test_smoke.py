"""Trivial but real tests so CI has something to run.

These exercise the model, the AUC metric, the per-node data generator, and the
core collaboration-lift claim — all without a Docker daemon or a live round.
"""

import numpy as np
import torch

from common.data import (
    BANK_IDS,
    FRAUD_TYPOLOGIES,
    prepare_dataset,
    split_dataset,
)
from common.metrics import roc_auc
from common.model import N_FEATURES, TabularNet, get_parameters, set_parameters


def test_model_forward_shape():
    model = TabularNet()
    out = model(torch.randn(4, N_FEATURES))
    assert out.shape == (4, 2)


def test_parameter_roundtrip():
    src, dst = TabularNet(), TabularNet()
    set_parameters(dst, get_parameters(src))
    for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
        assert torch.allclose(a, b)


def test_roc_auc_basics():
    # Perfectly separable -> 1.0; reversed -> 0.0; chance ~ 0.5.
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all ties -> chance
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))  # one class -> NaN


def test_each_bank_gets_its_own_isolated_slice(tmp_path):
    Xa, ya = prepare_dataset(str(tmp_path / "bank-a"), "bank-a")
    Xb, _ = prepare_dataset(str(tmp_path / "bank-b"), "bank-b")
    assert Xa.shape[1] == N_FEATURES
    assert not np.array_equal(Xa, Xb)
    # Re-loading from the same volume is stable.
    Xa2, _ = prepare_dataset(str(tmp_path / "bank-a"), "bank-a")
    assert np.array_equal(Xa, Xa2)


def test_non_iid_split_is_skewed():
    data = split_dataset()
    banks = data["banks"]
    # Every bank has data and at least one fraud example.
    for b in BANK_IDS:
        assert len(banks[b]["y"]) > 0
        assert banks[b]["y"].sum() > 0
    # Structuring concentrates in bank-a, layering in bank-b (non-IID).
    def frac(bank, typ):
        typ_arr = banks[bank]["typ"]
        return (typ_arr == typ).sum()
    assert frac("bank-a", "structuring") > frac("bank-c", "structuring")
    assert frac("bank-b", "layering") > frac("bank-c", "layering")
    # The shared held-out test set covers every typology.
    test_typ = data["test"]["typ"]
    for t in FRAUD_TYPOLOGIES:
        assert (test_typ == t).sum() > 0


def test_federation_beats_best_solo():
    # The core product claim, locked into CI on a small, fast config.
    from benchmark.core import BenchmarkConfig, run_benchmark

    res = run_benchmark(BenchmarkConfig(rounds=6, local_epochs=4, lr=0.05, seed=0))
    assert res["federated_overall"] > res["best_solo_overall"]
