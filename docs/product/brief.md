# Product Brief — Federated AML Network

> Engineering decisions live in [`docs/adr/`](../adr/). This brief is the product
> counterpart: who it's for, what success looks like, and what we deliberately
> don't build.

## The problem
Money-laundering typologies — structuring, layering, mule rings — are inherently
**cross-institutional**: the clearest signal lives in the data no single bank can
see. But banks can't pool raw transactions (GDPR, banking secrecy, FinCEN). So each
bank runs a siloed model that drowns analysts in false-positive SARs while real
rings slip through the gaps *between* institutions.

## Who it's for
- **Primary user:** the bank's financial-crime data-science team / MLRO who owns
  detection quality and the SAR backlog.
- **Economic buyer:** the head of financial crime / compliance — or a neutral
  consortium operator or regulator-backed utility that runs the network.
- **This is a consortium product** — its value compounds with every bank that joins.

## Job to be done
"Help my bank catch cross-institutional laundering rings we can't see alone —
without exposing our customers' data or breaching privacy law."

## What success looks like
- **North Star — collaboration lift:** uplift in true-positive detection of
  *cross-institutional* typologies (e.g. mule rings) versus a bank's solo model, at
  a fixed false-positive budget. If joining the network doesn't beat going alone,
  nothing else matters.
- **Value metrics:** false-positive reduction (analyst-hours saved) · cross-bank
  ring-detection rate · time-to-detect.
- **Quality / privacy metrics:** the ε privacy budget (made explicit) ·
  membership-inference attack success rate (must fall to ~baseline with DP) · model
  AUC under realistic non-IID data.

## Non-goals (what we deliberately don't build)
- **Never pool raw data.** Only clipped, noised model updates cross a boundary —
  this is the entire premise, not a feature flag.
- **Not a SAR-filing / case-management system.** We produce signals; we integrate
  with the bank's existing AML case tooling.
- **Not protection against a malicious aggregation server.** FedAvg + DP defends
  against reconstruction and membership inference, *not* a hostile server reading
  updates — secure aggregation / SMPC is an honest, stated boundary
  ([ADR 0001](../adr/0001-framework-and-aggregation.md)).
- **Not inline transaction blocking** — it's a detection/scoring model, not a
  real-time payment gate.
- **Not a replacement** for the bank's existing rules engine — it augments it.

## Sequencing — prove the riskiest assumption first
The riskiest *product* bet is the privacy/utility tradeoff, in two parts:

1. **Utility:** a federated model trained across non-IID bank data beats a solo
   baseline at catching cross-bank rings. *(Prove collaboration is worth it.)*
2. **Privacy is real:** the membership-inference attack succeeds without DP and
   **fails with DP enabled**, with ε stated.
3. Only then: orchestration polish, more nodes, CI.

If (1) and (2) don't hold, the network has no reason to exist — so they come first.

## Key risks & assumptions
- **Consortium cold-start & governance (biggest risk, and it's not technical).**
  Banks won't join on trust alone: who operates the network, who's liable, how is it
  governed? Adoption is a legal/trust problem. Mitigation: a neutral or
  regulator-blessed operator and a 2-bank pilot under a data-sharing agreement
  before opening up.
- **Privacy/utility knife-edge.** Too much noise → useless model; too little →
  leakage. ε must be defensible to a DPO *and* a regulator.
- **Contribution asymmetry / free-riding.** A data-poor bank benefits from others'
  data — the network needs an incentive/fairness story.
- **Regulatory acceptance.** Will a regulator accept a federated model's output in a
  SAR decision? Explainability of flags matters for adoption.

## The demo, framed as a user outcome
A mule ring moves funds A→B→C across three banks. Each bank's solo model sees only
one hop and flags nothing. The federated model — trained across all three, with no
bank ever seeing another's data — flags the ring. Then an attacker runs membership
inference against the shared model and **fails**: the privacy budget held.
*Catch what you can't see alone, without seeing each other's data.*
