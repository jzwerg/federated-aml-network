"""Bank client: train locally on this node's own data, join the FedAvg round.

Only model parameters cross the network boundary; raw transactions never leave
the node's ``/data`` volume. When ``DP_ENABLED`` is set, local training uses
Opacus DP-SGD (gradient clipping + Gaussian noise) and reports the ε budget spent.
"""

import logging
import os
import time

import flwr as fl
import torch
import torch.nn as nn

from common.data import load_tensors
from common.metrics import roc_auc
from common.model import TabularNet, fraud_scores, get_parameters, set_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [client] %(message)s")
log = logging.getLogger("client")

LOCAL_EPOCHS = 5
LEARNING_RATE = 0.05
VAL_FRACTION = 0.2


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _train_val_split(X: torch.Tensor, y: torch.Tensor, bank_id: str):
    """Deterministic per-bank split so evaluate() reports held-out (not in-sample) AUC."""
    n_val = max(1, int(len(X) * VAL_FRACTION))
    seed = sum(ord(c) for c in bank_id)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class BankClient(fl.client.NumPyClient):
    def __init__(self, bank_id: str, X: torch.Tensor, y: torch.Tensor):
        self.bank_id = bank_id
        self.model = TabularNet()
        self.X_train, self.y_train, self.X_val, self.y_val = _train_val_split(
            X, y, bank_id
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Differential privacy setup (lazy import so the non-DP path stays light).
        self.dp_enabled = _env_bool("DP_ENABLED", True)
        self.privacy_engine = None
        self.dp_cfg = None
        self.noise_multiplier = None
        self.dp_loader = None
        if self.dp_enabled:
            from opacus import PrivacyEngine
            from client.dp import DPConfig, compute_noise_multiplier, make_loader

            num_rounds = int(os.environ.get("NUM_ROUNDS", "3"))
            self.dp_cfg = DPConfig(
                target_epsilon=float(os.environ.get("EPSILON", "5.0"))
            )
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.dp_loader = make_loader(self.X_train, self.y_train, self.dp_cfg)
            self.noise_multiplier = compute_noise_multiplier(
                self.dp_cfg,
                n_train=len(self.X_train),
                total_epochs=num_rounds * LOCAL_EPOCHS,
            )
            log.info(
                "[%s] DP on: target epsilon=%.2f, sigma=%.3f, clip=%.2f",
                bank_id,
                self.dp_cfg.target_epsilon,
                self.noise_multiplier,
                self.dp_cfg.max_grad_norm,
            )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        if self.dp_enabled:
            return self._fit_dp(parameters)
        set_parameters(self.model, parameters)
        return self._fit_plain()

    def _fit_plain(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.model.train()
        loss = torch.tensor(0.0)
        for _ in range(LOCAL_EPOCHS):
            optimizer.zero_grad()
            loss = self.loss_fn(self.model(self.X_train), self.y_train)
            loss.backward()
            optimizer.step()
        log.info("[%s] local fit done, loss=%.4f", self.bank_id, loss.item())
        return (
            get_parameters(self.model),
            len(self.X_train),
            {"loss": float(loss.item())},
        )

    def _fit_dp(self, parameters):
        from client.dp import train_round

        # A fresh model each round: Opacus adds grad-sample hooks to the module it
        # wraps, so the same object can't be re-wrapped. The PrivacyEngine (and its
        # accountant) persists on the client, so ε still accumulates across rounds.
        model = TabularNet()
        set_parameters(model, parameters)
        loss, epsilon = train_round(
            self.privacy_engine,
            model,
            self.dp_loader,
            self.dp_cfg,
            self.noise_multiplier,
            LOCAL_EPOCHS,
            LEARNING_RATE,
        )
        log.info(
            "[%s] DP fit done, loss=%.4f, epsilon spent=%.3f", self.bank_id, loss, epsilon
        )
        return (
            get_parameters(model),
            len(self.X_train),
            {"loss": float(loss), "epsilon": float(epsilon)},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        with torch.no_grad():
            loss = self.loss_fn(self.model(self.X_val), self.y_val)
        auc = roc_auc(self.y_val.cpu().numpy(), fraud_scores(self.model, self.X_val))
        log.info("[%s] local eval: loss=%.4f auc=%.4f", self.bank_id, loss.item(), auc)
        return float(loss.item()), len(self.X_val), {"auc": float(auc)}


def main() -> None:
    bank_id = os.environ.get("BANK_ID", "bank-unknown")
    server_address = os.environ.get("SERVER_ADDRESS", "server:8080")
    data_dir = os.environ.get("DATA_DIR", "/data")

    X, y = load_tensors(data_dir, bank_id)
    log.info("[%s] loaded %d local samples from %s", bank_id, len(X), data_dir)

    client = BankClient(bank_id, X, y).to_client()

    # Retry until the server is accepting connections (it may still be starting).
    max_attempts = 30
    for attempt in range(1, max_attempts + 1):
        try:
            log.info(
                "[%s] connecting to %s (attempt %d/%d)",
                bank_id,
                server_address,
                attempt,
                max_attempts,
            )
            fl.client.start_client(server_address=server_address, client=client)
            log.info("[%s] federated session finished", bank_id)
            return
        except Exception as exc:  # noqa: BLE001 - retry on any connection failure
            log.warning("[%s] server not ready (%s); retrying in 2s", bank_id, exc)
            time.sleep(2)

    raise SystemExit(f"[{bank_id}] could not reach the Flower server")


if __name__ == "__main__":
    main()
