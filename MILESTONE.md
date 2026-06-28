# Milestone 0 — First `docker compose up`

The single goal of this milestone: **`make up` runs at least one federated round**
across a server and three isolated bank nodes that never share raw data. This is
*not* the membership-inference attack/defense demo (that's next). Keep scope tight.

## Definition of done
- `cp .env.example .env && make up` boots the `server` and three bank clients
  (`bank-a/b/c`), each needing a **Python Dockerfile**. The server waits for
  `MIN_CLIENTS=3`, then at least one FedAvg round completes.
- A metrics endpoint is reachable on **:8200** reporting the round number / basic
  metrics.
- Each bank reads only its **own** data volume — no shared raw-data volume.
- `make down` tears everything down cleanly.

## Smoke check (how you know it worked)
- `server` logs show `round 1` completing across 3 clients.
- `curl -fsS localhost:8200/metrics` (or `/health`) returns round/metric data.
- `make ps` shows the server + three bank services up.

## Explicitly out of scope (later milestones)
The membership-inference attack, careful DP/ε tuning, a realistic non-IID split, the
fraud-typology synthetic generator depth. A trivial model + tiny synthetic data is
fine here. See `PLAN.md`.

## Stack gotchas
- Use **CPU-only PyTorch wheels** to keep the image small and avoid GPU assumptions.
- Clients must wait for the server to be ready (retry the connection).
- Enforce isolation: each bank service has its **own** volume; no shared mount for
  raw transactions — this is the project's core claim, not a detail.
- `MIN_CLIENTS=3` so the server blocks until all three banks join.
- Use Python `3.12` to match CI.

## Shared conventions (portfolio-wide — keep identical across all four repos)
- **Branch:** `claude/product-thinking-repos-cmbegm`.
- **Task interface:** `make up` / `down` / `demo` / `test` / `logs`.
- **First boot needs no secrets** — `.env.example` defaults must boot.
- **Compose v2:** `docker compose` (space), not the deprecated `docker-compose`.
- **Host ports:** this project owns the **82xx** range.
- **Validate without a daemon:** `docker compose config -q` parses the stack even
  where Docker can't run (e.g. a Claude Code web session); a real boot must be
  verified on a machine with a Docker daemon.

## Paste-ready session kickoff
> Get this repo to its first `docker compose up` state per `MILESTONE.md`. Add a
> Python Dockerfile (CPU-only torch), a Flower server entrypoint that waits for 3
> clients and runs one FedAvg round, and a client entrypoint that trains on its own
> isolated synthetic data. Expose a metrics endpoint on the server. Don't build the
> attack demo, DP tuning, or a realistic non-IID split yet. Validate with
> `docker compose config -q`, then confirm the smoke check. Commit to
> `claude/product-thinking-repos-cmbegm` and push.
