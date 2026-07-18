# Roadmap — getting the project off the ground

`MILESTONE.md` was the contract for **Milestone 0** (first `docker compose up`),
which is now done: the stack boots, three isolated banks complete a FedAvg round,
and metrics are served on `:8200`. This file is the execution-oriented companion to
`PLAN.md` — it sequences the milestones that turn the working skeleton into the
product the brief describes, each scoped as its own session + commit in the same
tight style as M0.

## Sequencing principle (from `docs/product/brief.md`)

> Prove the riskiest bet first: **(1) utility** — a federated model beats a solo
> baseline at catching cross-bank rings; **(2) privacy is real** — the
> membership-inference attack succeeds without DP and **fails with DP on**, ε
> stated. *Only then* polish.

This also matches the natural dependency chain: you can't run a meaningful attack
until there's a model worth attacking, and you can't show "collaboration lift"
until the data has genuine cross-bank structure. So the milestones stack cleanly.

Status note: as of M0 the README still reads "Planning phase." Flip it to reflect
the working stack as part of M1.

---

## M1 — Real data + the utility story  ✅ done  *(highest value; everything depends on it)*

Replace the random-noise placeholder data with synthetic transactions that carry
the signal the product is about, and prove that federating actually helps.

### Definition of done
- A synthetic transaction generator (grow `common/data.py` or add `common/synth.py`):
  time-series transactions with injected typologies — **structuring**, **layering**,
  and a **mule ring that spans A→B→C** so the laundering signal genuinely lives
  *between* banks, not inside any one.
- A **non-IID split** — each bank sees a different slice of the distribution.
- A real (still small) PyTorch tabular classifier with a held-out evaluation set.
- A `make benchmark` target reporting **collaboration lift**: federated AUC vs. each
  bank's solo-model AUC on the cross-bank ring. The federated model must win.
- `/metrics` reports model quality (AUC) per round, not just loss.

### Out of scope
Differential privacy, the attack, hyperparameter tuning beyond "it converges."

### Smoke check
- `make up` still completes its round(s) on the new data.
- `make benchmark` prints federated AUC > best solo AUC on the mule-ring split.

---

## M2 — Differential privacy layer  ✅ done

Make the privacy guarantee real and measurable, using the env vars already plumbed
through compose (`DP_ENABLED`, `EPSILON`).

*Landed:* Opacus DP-SGD on each client (`client/dp.py`), gated by `DP_ENABLED`;
`EPSILON` treated as the target budget for the run, with the noise multiplier
derived by Opacus and the cumulative ε accounted across rounds and reported on
`:8200/metrics` (`epsilon_spent` / `target_epsilon`). Validated: with DP on the
model still learns (mean val AUC ~0.87) while ε converges to ~5.0 — graceful
degradation. `DP_ENABLED=false` reproduces the M1 vanilla path.

### Definition of done
- Opacus gradient clipping + Gaussian noise applied on the **client** during local
  training, gated by `DP_ENABLED`.
- The **ε privacy budget per round** is computed and exposed on `/metrics`.
- With DP on, utility degrades gracefully — the model is noisier, not broken.

### Out of scope
The attack itself; careful ε optimization (pick a defensible value and state it).

### Smoke check
- `DP_ENABLED=true make up` trains and `/metrics` shows a stated ε.
- `DP_ENABLED=false` reproduces M1 behaviour.

---

## M3 — The headline attack/defense demo  ⬅️ NEXT  *(what the README promises)*

### Definition of done
- A membership-inference attack against the shared model.
- `make demo` runs it both ways and shows the attack **succeeds against vanilla
  FedAvg and fails with DP enabled**, printing the attack success rate and ε.
- Wire it into CI — the `attack-demo` job in `.github/workflows/ci.yml` is already
  stubbed for exactly this.

### Out of scope
Defending against a *malicious aggregation server* — an explicitly stated boundary
(see `docs/adr/0001-framework-and-aggregation.md`).

### Smoke check
- `make demo` prints: attack success ≈ high (no DP) → ≈ baseline (DP on), with ε.
- CI's `attack-demo` job goes green on the real assertion, not the placeholder.

---

## M4 — Polish

- ADR documenting the DP mechanism and the ε choice (defensible to a DPO/regulator).
- Bake the collaboration-lift + attack results into CI as the real proof.
- Terminal recording of the demo; flip README `Status` from "Planning" to live.
- Consider scaling past three nodes and a contribution/fairness story (brief risks).

---

## Milestone → `PLAN.md` mapping

| Roadmap | `PLAN.md` milestone(s) |
| --- | --- |
| M0 (done) | 3 — Federated orchestration (skeleton) |
| M1 (done) | 1 — Synthetic generator · 2 — Local model + training |
| M2 (done) | 4 — Differential privacy layer |
| M3 (next) | 5 — Attack demo |
| M4 | 6 — Polish |
