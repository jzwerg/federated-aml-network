"""A trivial tabular classifier shared across nodes.

This is deliberately small — the milestone only needs *a* model that can train
and have its weights averaged. The realistic fraud-typology model arrives in a
later milestone (see PLAN.md).
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

# Number of input features in the synthetic transaction vectors.
N_FEATURES = 8
N_CLASSES = 2  # fraud / not-fraud


class TabularNet(nn.Module):
    """Small MLP binary classifier over tabular transaction features."""

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
