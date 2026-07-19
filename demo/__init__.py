"""Membership-inference attack/defense demo.

Run via ``python -m demo`` (or ``make demo``). Trains the shared model two ways —
vanilla FedAvg and DP-SGD FedAvg — then runs a loss-based membership-inference
attack against each. The headline: the attack *succeeds* against the vanilla model
and *fails* (falls to ~chance) once differential privacy is on, with the ε budget
reported.

Threat-model boundary (see docs/adr/0001): DP defends against membership inference
and reconstruction, not against a malicious aggregation server reading updates.
"""
