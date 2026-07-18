"""Differential-privacy path tests.

Verify that DP-SGD trains, spends a bounded ε budget, and that the ε accountant
accumulates across rounds (cumulative, not per-round).
"""

import numpy as np
import torch

from client.dp import DPConfig, compute_noise_multiplier, make_loader, train_round
from common.model import N_FEATURES, TabularNet, get_parameters


def _toy_data(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, N_FEATURES, generator=g)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X, y


def test_noise_multiplier_is_positive():
    cfg = DPConfig(target_epsilon=5.0, batch_size=32)
    sigma = compute_noise_multiplier(cfg, n_train=256, total_epochs=4)
    assert sigma > 0


def test_dp_round_trains_and_bounds_epsilon():
    torch.manual_seed(0)
    X, y = _toy_data()
    cfg = DPConfig(target_epsilon=5.0, batch_size=32)
    # noise sized for exactly the steps we run -> ε ends near the target.
    sigma = compute_noise_multiplier(cfg, n_train=len(X), total_epochs=2)

    from opacus import PrivacyEngine

    model = TabularNet()
    loader = make_loader(X, y, cfg)
    # copy: get_parameters shares storage with the model tensors (via .numpy()).
    before = [p.copy() for p in get_parameters(model)]
    loss, eps = train_round(PrivacyEngine(accountant="rdp"), model, loader, cfg, sigma,
                            local_epochs=2, lr=0.05)
    assert np.isfinite(loss)
    assert 0.0 < eps <= cfg.target_epsilon + 0.5
    after = get_parameters(model)
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_epsilon_accumulates_across_rounds():
    torch.manual_seed(0)
    X, y = _toy_data()
    cfg = DPConfig(target_epsilon=5.0, batch_size=32)
    sigma = compute_noise_multiplier(cfg, n_train=len(X), total_epochs=4)

    from opacus import PrivacyEngine

    # One persistent engine/accountant + loader, a fresh model each round.
    engine = PrivacyEngine(accountant="rdp")
    loader = make_loader(X, y, cfg)
    m1 = TabularNet()
    _, eps1 = train_round(engine, m1, loader, cfg, sigma, local_epochs=2, lr=0.05)
    m2 = TabularNet()
    _, eps2 = train_round(engine, m2, loader, cfg, sigma, local_epochs=2, lr=0.05)
    assert eps2 > eps1  # same accountant -> cumulative budget grows
