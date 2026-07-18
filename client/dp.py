"""Differential privacy for the bank clients (Opacus DP-SGD).

Gated by ``DP_ENABLED``. When on, each client trains with per-sample gradient
clipping + calibrated Gaussian noise, and the privacy budget (ε) is accounted
across rounds so the value reported on ``/metrics`` is the *cumulative* budget
spent, not a per-round underestimate.

Design choice (see ROADMAP.md M2): ``EPSILON`` is treated as the target budget for
the whole federated run. Opacus derives the noise multiplier needed to hit that
target over the planned number of local optimization steps; the same multiplier is
reused each round and the accountant accumulates. Careful ε optimization and the
privacy/utility knife-edge are out of scope for this milestone — ε=5.0 is a
common, defensible "moderate privacy" default.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier


def make_loader(X: torch.Tensor, y: torch.Tensor, cfg: "DPConfig") -> DataLoader:
    """Build the per-client loader once and reuse it every round.

    Reusing the same dataset object keeps Opacus's per-dataset privacy accounting
    happy (and quiet) across rounds.
    """
    return DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True)


@dataclass
class DPConfig:
    target_epsilon: float = 5.0
    target_delta: float = 1e-5     # convention: delta < 1/n_train
    max_grad_norm: float = 1.0     # per-sample gradient clipping bound (C)
    batch_size: int = 64


def compute_noise_multiplier(cfg: DPConfig, n_train: int, total_epochs: int) -> float:
    """Noise multiplier (σ) that hits ``target_epsilon`` over the full run."""
    sample_rate = min(1.0, cfg.batch_size / n_train)
    return get_noise_multiplier(
        target_epsilon=cfg.target_epsilon,
        target_delta=cfg.target_delta,
        sample_rate=sample_rate,
        epochs=total_epochs,
        accountant="rdp",
    )


def train_round(
    privacy_engine: PrivacyEngine,
    model: nn.Module,
    loader: DataLoader,
    cfg: DPConfig,
    noise_multiplier: float,
    local_epochs: int,
    lr: float,
) -> tuple[float, float]:
    """One round of DP-SGD local training.

    ``model`` is trained in place (its parameters are shared with the Opacus
    wrapper), so the caller reads updated weights straight off ``model``. Pass a
    fresh ``model`` each round (Opacus can't re-wrap one) but the SAME
    ``privacy_engine`` and ``loader`` so ε accounting accumulates. Returns
    ``(last_loss, cumulative_epsilon)``.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Reusing the same PrivacyEngine across rounds keeps one accountant, so the
    # reported ε is the total budget spent so far.
    priv_model, priv_optimizer, priv_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=cfg.max_grad_norm,
    )

    loss_fn = nn.CrossEntropyLoss()
    priv_model.train()
    last_loss = 0.0
    for _ in range(local_epochs):
        for xb, yb in priv_loader:
            priv_optimizer.zero_grad()
            loss = loss_fn(priv_model(xb), yb)
            loss.backward()
            priv_optimizer.step()
            last_loss = float(loss.item())

    epsilon = privacy_engine.get_epsilon(cfg.target_delta)
    return last_loss, float(epsilon)
