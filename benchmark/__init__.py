"""Collaboration-lift benchmark: federated model vs. each bank's solo model.

Run via ``python -m benchmark`` (or ``make benchmark``). Pure local computation —
a faithful in-process FedAvg simulation, no Docker daemon or gRPC required — so it
also runs in CI.
"""
