"""Flower server: wait for MIN_CLIENTS banks, run FedAvg, expose metrics on :8000.

Aggregates FedAvg across the isolated bank clients and reports round number, mean
local AUC, and — when DP is enabled — the ε privacy budget spent. The
membership-inference attack demo is the next milestone (see ROADMAP.md M3).
"""

import logging
import os
import threading

import flwr as fl
import uvicorn
from fastapi import FastAPI


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
log = logging.getLogger("server")

# Shared, in-process metrics — written by the strategy, read by the HTTP endpoint.
METRICS = {
    "rounds_completed": 0,
    "num_rounds_target": int(os.environ.get("NUM_ROUNDS", "3")),
    "min_clients": int(os.environ.get("MIN_CLIENTS", "3")),
    "dp_enabled": _env_bool("DP_ENABLED", True),
    "target_epsilon": float(os.environ.get("EPSILON", "5.0")),
    "latest_loss": None,
    "latest_auc": None,
    "epsilon_spent": None,
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
        # ε is a per-client budget; report the max (worst-case budget spent).
        epsilons = [
            r.metrics["epsilon"]
            for _, r in results
            if r.metrics and "epsilon" in r.metrics
        ]
        if epsilons:
            METRICS["epsilon_spent"] = max(epsilons)
        log.info(
            "round %d: aggregated fit across %d client(s)%s",
            server_round,
            len(results),
            f", epsilon spent={METRICS['epsilon_spent']:.3f}"
            if METRICS["epsilon_spent"] is not None
            else "",
        )
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        aucs = [
            r.metrics["auc"]
            for _, r in results
            if r.metrics and "auc" in r.metrics
        ]
        if aucs:
            METRICS["latest_auc"] = sum(aucs) / len(aucs)
            log.info("round %d: mean local AUC %.4f", server_round, METRICS["latest_auc"])
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
