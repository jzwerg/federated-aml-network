"""The shared tabular fraud/AML classifier.

A small MLP over the synthetic transaction features defined in ``common.data``.
Every node trains an identical architecture so FedAvg can average weights. The DP
layer and the membership-inference attack are still later milestones (see
``ROADMAP.md``).
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

# Number of input features in the synthetic transaction vectors
# (kept in sync with the feature schema in ``common.data``).
N_FEATURES = 12
N_CLASSES = 2  # fraud / not-fraud


class TabularNet(nn.Module):
    """Small MLP binary classifier over tabular transaction features."""

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fraud_scores(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Per-row fraud score (logit margin for the fraud class) as a NumPy array.

    AUC is invariant to monotonic transforms, so the raw class-1 vs class-0 logit
    margin is a fine ranking score — no softmax needed.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        margin = logits[:, 1] - logits[:, 0]
    return margin.cpu().numpy()


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Flower-compatible: model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays (from FedAvg) back into the model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(np.asarray(v)) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
