"""Core of the membership-inference attack/defense demo.

Attack: a standard loss-based membership-inference attack. A record the model
trained on ("member") tends to get a *lower* loss than a fresh record from the
same distribution ("non-member"). The attacker scores each record by ``-loss`` and
we measure how well that separates members from non-members (ROC AUC, plus
best-threshold accuracy). AUC ~0.5 means no leakage; well above 0.5 means the model
memorized its members.

Why label noise: with tiny, easily-separable synthetic data a well-generalizing
model gives members and non-members nearly identical loss, so there's little to
leak. Injecting label noise into the population (a standard MIA setup) forces the
vanilla model to *memorize* individual training labels to drive training loss down
— creating a real member/non-member gap. DP-SGD bounds any single example's
influence, so it cannot memorize those labels and the gap collapses. That collapse
— not the absolute numbers — is the result.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.core import train_local, weighted_average
from common.data import BANK_IDS, split_dataset
from common.metrics import roc_auc
from common.model import TabularNet, fraud_scores, get_parameters, set_parameters


@dataclass
class DemoConfig:
    # Members must be few enough (relative to the model's ~1k params) that the
    # vanilla model can memorize them — that's what creates the leak to attack.
    members_per_bank: int = 100
    rounds: int = 12
    local_epochs: int = 25       # many epochs -> vanilla memorizes the noisy members
    lr: float = 0.1
    target_epsilon: float = 3.0  # a firm budget so DP clearly closes the gap
    max_grad_norm: float = 1.0
    batch_size: int = 64
    label_noise: float = 0.30    # fraction of labels flipped (amplifies the signal)
    seed: int = 0


def _flip_labels(y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    flipped = y.copy()
    mask = rng.random(len(y)) < noise
    flipped[mask] = 1 - flipped[mask]
    return flipped


def _prepare(cfg: DemoConfig):
    """Per-bank noisy training slices (members) + a noisy held-out set (non-members)."""
    data = split_dataset()
    rng = np.random.default_rng(cfg.seed)

    banks = {}
    for b in BANK_IDS:
        X = data["banks"][b]["X"]
        y = data["banks"][b]["y"]
        # Subsample a small member set so the model can memorize it.
        k = min(cfg.members_per_bank, len(X))
        idx = rng.choice(len(X), k, replace=False)
        X = X[idx]
        y = _flip_labels(y[idx], cfg.label_noise, rng)
        banks[b] = (torch.from_numpy(X), torch.from_numpy(y))

    test = data["test"]
    non_X = torch.from_numpy(test["X"])
    non_y = torch.from_numpy(_flip_labels(test["y"], cfg.label_noise, rng))
    clean_test_y = test["y"]  # for a utility read-out, scored against true labels
    return banks, (non_X, non_y), clean_test_y, test["X"]


def _train_vanilla(banks, cfg: DemoConfig):
    torch.manual_seed(cfg.seed)
    params = get_parameters(TabularNet())
    for _ in range(cfg.rounds):
        round_params, sizes = [], []
        for b in BANK_IDS:
            model = TabularNet()
            set_parameters(model, params)
            X, y = banks[b]
            train_local(model, X, y, cfg.local_epochs, cfg.lr)
            round_params.append(get_parameters(model))
            sizes.append(len(X))
        params = weighted_average(round_params, sizes)
    model = TabularNet()
    set_parameters(model, params)
    return model


def _train_dp(banks, cfg: DemoConfig):
    from opacus import PrivacyEngine
    from client.dp import DPConfig, compute_noise_multiplier, make_loader, train_round

    dp_cfg = DPConfig(
        target_epsilon=cfg.target_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        batch_size=cfg.batch_size,
    )
    total_epochs = cfg.rounds * cfg.local_epochs
    engines, loaders, sigmas = {}, {}, {}
    for b in BANK_IDS:
        X, y = banks[b]
        engines[b] = PrivacyEngine(accountant="rdp")
        loaders[b] = make_loader(X, y, dp_cfg)
        sigmas[b] = compute_noise_multiplier(dp_cfg, n_train=len(X), total_epochs=total_epochs)

    torch.manual_seed(cfg.seed)
    params = get_parameters(TabularNet())
    epsilon = 0.0
    for _ in range(cfg.rounds):
        round_params, sizes = [], []
        for b in BANK_IDS:
            model = TabularNet()
            set_parameters(model, params)
            _, epsilon_b = train_round(
                engines[b], model, loaders[b], dp_cfg, sigmas[b], cfg.local_epochs, cfg.lr
            )
            round_params.append(get_parameters(model))
            sizes.append(len(banks[b][1]))
            epsilon = max(epsilon, epsilon_b)
        params = weighted_average(round_params, sizes)
    model = TabularNet()
    set_parameters(model, params)
    return model, float(epsilon)


def _per_example_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        losses = F.cross_entropy(model(X), y, reduction="none")
    return losses.cpu().numpy()


def _best_threshold_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Max accuracy of predicting member when score >= t, over all thresholds."""
    best = 0.0
    for t in np.unique(scores):
        acc = ((scores >= t).astype(int) == labels).mean()
        best = max(best, acc)
    return float(best)


def membership_attack(model, members, non_members, seed: int) -> Dict[str, float]:
    """Loss-based MIA. Returns attack AUC and best-threshold accuracy on a balanced set."""
    mem_X, mem_y = members
    non_X, non_y = non_members
    loss_m = _per_example_loss(model, mem_X, mem_y)
    loss_n = _per_example_loss(model, non_X, non_y)

    # Balance the two groups so accuracy is meaningful (chance = 0.5).
    rng = np.random.default_rng(seed)
    n = min(len(loss_m), len(loss_n))
    loss_m = loss_m[rng.choice(len(loss_m), n, replace=False)]
    loss_n = loss_n[rng.choice(len(loss_n), n, replace=False)]

    scores = np.concatenate([-loss_m, -loss_n])  # higher score => more member-like
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return {
        "auc": roc_auc(labels, scores),
        "accuracy": _best_threshold_accuracy(labels, scores),
        "n_per_group": n,
    }


def run_demo(config: DemoConfig | None = None) -> Dict[str, object]:
    """Train vanilla + DP models and attack both. Pure computation, no printing."""
    cfg = config or DemoConfig()
    banks, non_members, clean_test_y, clean_test_X = _prepare(cfg)

    # Members = the union of the banks' (noisy) training rows.
    mem_X = torch.cat([banks[b][0] for b in BANK_IDS])
    mem_y = torch.cat([banks[b][1] for b in BANK_IDS])
    members = (mem_X, mem_y)

    vanilla = _train_vanilla(banks, cfg)
    dp_model, epsilon = _train_dp(banks, cfg)

    clean_X_t = torch.from_numpy(clean_test_X)

    def utility(model):  # AUC against the TRUE (un-noised) test labels
        return roc_auc(clean_test_y, fraud_scores(model, clean_X_t))

    return {
        "config": cfg,
        "vanilla_attack": membership_attack(vanilla, members, non_members, cfg.seed),
        "dp_attack": membership_attack(dp_model, members, non_members, cfg.seed),
        "epsilon_spent": epsilon,
        "vanilla_utility_auc": utility(vanilla),
        "dp_utility_auc": utility(dp_model),
    }
