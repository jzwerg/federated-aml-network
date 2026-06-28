"""Bank client: train locally on this node's own data, join the FedAvg round.

Only model parameters cross the network boundary; raw transactions never leave
the node's ``/data`` volume. Differential privacy is a later milestone — the
DP_ENABLED / EPSILON env vars are accepted now but not yet applied.
"""

import logging
import os
import time

import flwr as fl
import torch
import torch.nn as nn

from common.data import load_tensors
from common.model import TabularNet, get_parameters, set_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [client] %(message)s")
log = logging.getLogger("client")

LOCAL_EPOCHS = 3
LEARNING_RATE = 0.05


class BankClient(fl.client.NumPyClient):
    def __init__(self, bank_id: str, X: torch.Tensor, y: torch.Tensor):
        self.bank_id = bank_id
        self.model = TabularNet()
        self.X = X
        self.y = y
        self.loss_fn = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.model.train()
        loss = torch.tensor(0.0)
        for _ in range(LOCAL_EPOCHS):
            optimizer.zero_grad()
            loss = self.loss_fn(self.model(self.X), self.y)
            loss.backward()
            optimizer.step()
        log.info("[%s] local fit done, loss=%.4f", self.bank_id, loss.item())
        return get_parameters(self.model), len(self.X), {"loss": float(loss.item())}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X)
            loss = self.loss_fn(logits, self.y)
            acc = (logits.argmax(dim=1) == self.y).float().mean().item()
        return float(loss.item()), len(self.X), {"accuracy": acc}


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
