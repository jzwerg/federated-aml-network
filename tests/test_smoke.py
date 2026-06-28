"""Trivial but real smoke tests so CI has something to run.

These exercise the actual model and per-node data code without needing a Docker
daemon or a live federated round.
"""

import numpy as np
import torch

from common.data import prepare_dataset
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


def test_each_bank_gets_its_own_data(tmp_path):
    # Distinct banks -> distinct (isolated) datasets, deterministic per bank.
    Xa, ya = prepare_dataset(str(tmp_path / "bank-a"), "bank-a")
    Xb, _ = prepare_dataset(str(tmp_path / "bank-b"), "bank-b")
    assert Xa.shape[1] == N_FEATURES
    assert not np.array_equal(Xa, Xb)
    # Re-loading from the same volume is stable.
    Xa2, _ = prepare_dataset(str(tmp_path / "bank-a"), "bank-a")
    assert np.array_equal(Xa, Xa2)
