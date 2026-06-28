"""CLI: run the collaboration-lift benchmark, print the table, assert the claim.

Exits non-zero if the federated model fails to beat the best solo model, so
``make benchmark`` and CI fail loudly if the core product claim ever regresses.
"""

import sys

from common.data import BANK_IDS, FRAUD_TYPOLOGIES
from benchmark.core import BenchmarkConfig, run_benchmark


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> int:
    cfg = BenchmarkConfig()
    res = run_benchmark(cfg)

    cols = ["overall"] + FRAUD_TYPOLOGIES
    header = "model".ljust(14) + "".join(c[:11].rjust(13) for c in cols)
    print("\nCollaboration lift — ROC AUC on the shared held-out test set")
    print(f"(FedAvg: {cfg.rounds} rounds x {cfg.local_epochs} local epochs)\n")
    print(header)
    print("-" * len(header))

    for b in BANK_IDS:
        n = res["sample_counts"][b]
        row = f"solo {b}".ljust(14)
        row += "".join(_fmt(res["solo"][b][c]).rjust(13) for c in cols)
        print(row + f"   (n={n})")

    fed_row = "federated".ljust(14) + "".join(
        _fmt(res["federated"][c]).rjust(13) for c in cols
    )
    print(fed_row)
    print("-" * len(header))

    lift = res["lift"]
    print(
        f"\nfederated overall {_fmt(res['federated_overall'])} vs. "
        f"best solo {_fmt(res['best_solo_overall'])}  ->  lift {lift:+.3f}"
    )

    if res["federated_overall"] > res["best_solo_overall"]:
        print("PASS: federation beats every bank's solo model.\n")
        return 0
    print("FAIL: federation did not beat the best solo model.\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
