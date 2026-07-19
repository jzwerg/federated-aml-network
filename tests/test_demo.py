"""Attack/defense claim, locked into CI on a small, fast config.

The membership-inference attack must succeed against the vanilla model and fall to
~chance once DP is on.
"""

from dataclasses import replace

from demo.core import DemoConfig, run_demo


def test_attack_succeeds_without_dp_and_fails_with_dp():
    cfg = replace(DemoConfig(), rounds=6, local_epochs=30)  # fast but shows the gap
    res = run_demo(cfg)
    vanilla = res["vanilla_attack"]["auc"]
    dp = res["dp_attack"]["auc"]

    assert vanilla > 0.55           # attack succeeds against vanilla FedAvg
    assert dp < 0.56                # attack falls to ~chance with DP
    assert vanilla - dp >= 0.03     # DP measurably reduces the leak
    assert res["epsilon_spent"] > 0  # a real budget was spent and reported
