"""CLI: run the attack/defense demo, print the headline, assert the claim.

Exits non-zero unless the membership-inference attack *succeeds* against the
vanilla model and *fails* (falls to ~chance) with DP enabled — so ``make demo``
and CI fail loudly if the privacy guarantee ever regresses.
"""

import sys

from demo.core import DemoConfig, run_demo

# The attack must clearly beat chance without DP, and collapse toward chance with
# DP. Thresholds carry margin over the observed seed-0 result (vanilla ~0.65,
# DP ~0.50).
VANILLA_MIN = 0.58   # attack must succeed without DP
DP_MAX = 0.58        # attack must fail (near chance) with DP
MIN_GAP = 0.05       # DP must measurably reduce the leak


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> int:
    cfg = DemoConfig()
    res = run_demo(cfg)
    v = res["vanilla_attack"]
    d = res["dp_attack"]
    gap = v["auc"] - d["auc"]

    print("\nMembership-inference attack — does the shared model leak who it trained on?")
    print(
        f"(FedAvg: {cfg.rounds} rounds x {cfg.local_epochs} local epochs; "
        f"{cfg.members_per_bank} members/bank; {int(cfg.label_noise*100)}% label noise)\n"
    )
    print("model            attack AUC   attack acc   verdict")
    print("-" * 56)
    print(
        f"vanilla FedAvg      {_fmt(v['auc'])}       {_fmt(v['accuracy'])}    "
        f"{'LEAKS (attack succeeds)' if v['auc'] > VANILLA_MIN else 'no clear leak'}"
    )
    print(
        f"DP-SGD (eps={res['epsilon_spent']:.2f})    {_fmt(d['auc'])}       {_fmt(d['accuracy'])}    "
        f"{'protected (attack ~chance)' if d['auc'] < DP_MAX else 'still leaks'}"
    )
    print("-" * 56)
    print(
        f"\nleak reduced by {gap:+.3f} AUC once DP is on  "
        f"(chance = 0.500; ε budget spent = {res['epsilon_spent']:.2f})"
    )
    print(
        f"utility retained — test AUC vs true labels: "
        f"vanilla {_fmt(res['vanilla_utility_auc'])}, DP {_fmt(res['dp_utility_auc'])}\n"
    )

    ok = v["auc"] > VANILLA_MIN and d["auc"] < DP_MAX and gap >= MIN_GAP
    if ok:
        print("PASS: attack succeeds without DP and fails once DP is enabled.\n")
        return 0
    print(
        "FAIL: expected the attack to succeed without DP and fail with DP.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
