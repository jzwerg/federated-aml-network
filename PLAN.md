# Build Plan — Federated AML Network

## Goal

Three simulated banks train a shared fraud/AML model via federated averaging, with no raw data leaving any node, and differential privacy bounding what the shared model can leak about any individual.

## Definition of done

`docker-compose up` trains a global model across 3 isolated bank nodes; a single command runs the membership-inference attack demo, showing it succeeds without DP and fails with DP enabled.

## Milestones

1. **Synthetic transaction generator** — time-series transactions with injected fraud typologies (structuring, layering, mule rings). Each bank receives a *different* (non-IID) distribution.
2. **Local model + training loop** — PyTorch tabular classifier, per-node training.
3. **Federated orchestration** — Flower server implementing FedAvg; Docker Compose with 1 server + 3 bank nodes; FastAPI metrics endpoint.
4. **Differential privacy layer** — gradient clipping + Gaussian noise via Opacus; expose the ε privacy budget per round.
5. **Attack demo** — membership-inference attack; succeeds against vanilla FedAvg, fails with DP.
6. **Polish** — README architecture diagram, ADRs, GitHub Actions CI with a meaningful test suite.

## Key technical challenges

- Keeping nodes genuinely isolated (separate containers, no shared volumes for raw data).
- Balancing the privacy/utility tradeoff — choosing ε so the model is still useful.
- Making the non-IID split realistic without making convergence impossible.

## Decisions captured

- **Flower over PySyft** — see `docs/adr/0001-framework-and-aggregation.md`.
- **FedAvg over SMPC** for this scale — see same ADR.
