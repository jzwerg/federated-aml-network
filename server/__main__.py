"""Flower server: wait for MIN_CLIENTS banks, run FedAvg, expose metrics on :8000.

Scope is the first federated round only. Differential privacy / ε tuning and the
attack demo are later milestones (see MILESTONE.md / PLAN.md).
"""

import logging
import os
import threading

import flwr as fl
import uvicorn
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
log = logging.getLogger("server")

# Shared, in-process metrics — written by the strategy, read by the HTTP endpoint.
METRICS = {
    "rounds_completed": 0,
    "num_rounds_target": int(os.environ.get("NUM_ROUNDS", "3")),
    "min_clients": int(os.environ.get("MIN_CLIENTS", "3")),
    "latest_loss": None,
    "latest_accuracy": None,
}

app = FastAPI(title="Federated AML Network — metrics")


@app.get("/health")
def health():
    return {"status": "ok", "rounds_completed": METRICS["rounds_completed"]}


@app.get("/metrics")
def metrics():
    return METRICS


class MetricsFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that records round number and client-reported metrics."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        METRICS["rounds_completed"] = server_round
        losses = [
            r.metrics["loss"]
            for _, r in results
            if r.metrics and "loss" in r.metrics
        ]
        if losses:
            METRICS["latest_loss"] = sum(losses) / len(losses)
        log.info(
            "round %d: aggregated fit across %d client(s)", server_round, len(results)
        )
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        accs = [
            r.metrics["accuracy"]
            for _, r in results
            if r.metrics and "accuracy" in r.metrics
        ]
        if accs:
            METRICS["latest_accuracy"] = sum(accs) / len(accs)
        return aggregated


def _run_metrics_server() -> None:
    port = int(os.environ.get("METRICS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def main() -> None:
    min_clients = METRICS["min_clients"]
    num_rounds = METRICS["num_rounds_target"]

    # Serve metrics in the background so it is reachable for the whole run.
    threading.Thread(target=_run_metrics_server, daemon=True).start()
    log.info("metrics endpoint on :%s", os.environ.get("METRICS_PORT", "8000"))

    strategy = MetricsFedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    log.info(
        "waiting for %d client(s); will run %d FedAvg round(s)",
        min_clients,
        num_rounds,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    log.info(
        "federated training complete: %d round(s) done", METRICS["rounds_completed"]
    )


if __name__ == "__main__":
    main()
