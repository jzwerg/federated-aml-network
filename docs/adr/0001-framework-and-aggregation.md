# ADR 0001 — Federated framework and aggregation strategy

- **Status:** Accepted
- **Context:** We need a federated learning framework and an aggregation strategy for a simulated multi-bank AML model.

## Decision

Use **Flower** as the federated learning framework and **Federated Averaging (FedAvg)** as the aggregation strategy, layered with **differential privacy** (gradient clipping + Gaussian noise via Opacus).

## Rationale

### Flower over PySyft
- Flower is actively maintained and the de-facto production-standard FL framework; PySyft has had significant API churn and a heavier dependency footprint.
- Flower is framework-agnostic (works cleanly with PyTorch) and has a simple client/server abstraction that maps directly to the "bank node / orchestration server" model.

### FedAvg over Secure Multiparty Computation (SMPC)
- For a portfolio-scale simulation, FedAvg demonstrates the core federated concept clearly and converges reliably.
- SMPC adds substantial cryptographic complexity and runtime cost that obscures the ML story without proportional learning value at this scale.
- We instead achieve the privacy guarantee through **differential privacy on model updates**, which is cheaper to run and produces a quantifiable (ε) privacy budget that we can demonstrate with an attack.

## Consequences

- The threat model is honest about its boundary: FedAvg + DP protects against data reconstruction and membership inference, but does not hide model updates from a malicious server. A follow-up ADR may revisit secure aggregation if we extend the threat model.
