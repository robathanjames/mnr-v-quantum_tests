"""A compact image encoder, differentiable quantum readout, and classifier."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .quantum import QuantumLayer, _integer


class SimpleCNN(nn.Module):
    """Extract 128 features by default from images of any height/width >= 4."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.in_channels = _integer("in_channels", in_channels)
        feature_dim = _integer("feature_dim", feature_dim)
        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
            raise TypeError("images must be a floating-point torch.Tensor")
        if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"images must have shape (nonempty batch, {self.in_channels}, height, width)"
            )
        if min(x.shape[-2:]) < 4:
            raise ValueError("image height and width must both be >= 4")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x).flatten(start_dim=1)
        return F.relu(self.fc1(x))


class HybridModel(nn.Module):
    """CNN -> bounded RY angles -> entangled Z features -> class logits.

    Defaults preserve the original 4-qubit, 10-class experiment. Pass logits
    directly to CrossEntropyLoss; apply softmax only when displaying probabilities.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        num_classes: int = 10,
        *,
        in_channels: int = 1,
        shots: int | None = None,
        seed: int | None = 7,
    ):
        super().__init__()
        num_classes = _integer("num_classes", num_classes, minimum=2)
        self.quantum = QuantumLayer(n_qubits, shots=shots, seed=seed)
        self.cnn = SimpleCNN(in_channels=in_channels)
        self.fc2 = nn.Linear(128, self.quantum.n_qubits)
        self.fc3 = nn.Linear(self.quantum.n_qubits, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        angles = math.pi * torch.tanh(self.fc2(features))
        # Keep both signs: ReLU would discard negative quantum expectation values.
        return self.fc3(self.quantum(angles))
