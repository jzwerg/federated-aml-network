"""Core of the collaboration-lift benchmark.

Trains one solo model per bank and one federated model (in-process FedAvg over the
same per-bank slices), then scores all of them on the shared held-out test set.
The FedAvg simulation mirrors what the Flower server does — weighted parameter
averaging across clients each round — but runs locally and deterministically.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from common.data import BANK_IDS, FRAUD_TYPOLOGIES, split_dataset
from common.metrics import roc_auc
from common.model import TabularNet, fraud_scores, get_parameters, set_parameters


@dataclass
class BenchmarkConfig:
    rounds: int = 8
    local_epochs: int = 5
    lr: float = 0.05
    seed: int = 0


def train_local(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                epochs: int, lr: float) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()


def weighted_average(params: List[List[np.ndarray]], sizes: List[int]) -> List[np.ndarray]:
    total = float(sum(sizes))
    return [
        sum(p[i] * (n / total) for p, n in zip(params, sizes))
        for i in range(len(params[0]))
    ]


def _auc_by_typology(model: nn.Module, test: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Overall AUC plus per-typology AUC (each typology vs. legit only)."""
    X = torch.from_numpy(test["X"])
    y = test["y"]
    typ = test["typ"]
    scores = fraud_scores(model, X)
    out = {"overall": roc_auc(y, scores)}
    legit = typ == "legit"
    for t in FRAUD_TYPOLOGIES:
        sel = legit | (typ == t)
        out[t] = roc_auc(y[sel], scores[sel])
    return out


def run_benchmark(config: BenchmarkConfig | None = None) -> Dict[str, object]:
    """Run the benchmark and return a results dict (no printing/asserting)."""
    cfg = config or BenchmarkConfig()
    data = split_dataset()
    test = data["test"]

    bank_tensors = {
        b: (torch.from_numpy(data["banks"][b]["X"]),
            torch.from_numpy(data["banks"][b]["y"]))
        for b in BANK_IDS
    }
    total_epochs = cfg.rounds * cfg.local_epochs

    # Solo models: each bank trains only on its own slice, for the same total
    # epochs the federated model gets, so the comparison is compute-fair.
    solo: Dict[str, Dict[str, float]] = {}
    for b in BANK_IDS:
        torch.manual_seed(cfg.seed)
        model = TabularNet()
        X, y = bank_tensors[b]
        train_local(model, X, y, total_epochs, cfg.lr)
        solo[b] = _auc_by_typology(model, test)

    # Federated model: FedAvg across the three slices.
    torch.manual_seed(cfg.seed)
    global_params = get_parameters(TabularNet())
    for _ in range(cfg.rounds):
        round_params, sizes = [], []
        for b in BANK_IDS:
            model = TabularNet()
            set_parameters(model, global_params)
            X, y = bank_tensors[b]
            train_local(model, X, y, cfg.local_epochs, cfg.lr)
            round_params.append(get_parameters(model))
            sizes.append(len(X))
        global_params = weighted_average(round_params, sizes)

    fed_model = TabularNet()
    set_parameters(fed_model, global_params)
    federated = _auc_by_typology(fed_model, test)

    best_solo = max(solo[b]["overall"] for b in BANK_IDS)
    return {
        "config": cfg,
        "solo": solo,
        "federated": federated,
        "best_solo_overall": best_solo,
        "federated_overall": federated["overall"],
        "lift": federated["overall"] - best_solo,
        "sample_counts": {b: int(len(bank_tensors[b][1])) for b in BANK_IDS},
    }
