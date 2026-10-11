from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


class MelCNN(nn.Module):
    """Configurable CNN for image-like log-Mel spectrograms."""

    def __init__(self, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = [int(value) for value in config["channels"]]
        kernel_sizes = _expand(config.get("kernel_sizes", 3), len(channels))
        pool_sizes = _expand(config.get("pool_sizes", 2), len(channels))
        dropout = float(config.get("dropout", 0.0))
        use_batch_norm = bool(config.get("batch_norm", True))
        hidden_dim = int(config.get("classifier_hidden_dim", 256))

        layers: list[nn.Module] = []
        in_channels = int(config.get("input_channels", 1))
        for out_channels, kernel_size, pool_size in zip(channels, kernel_sizes, pool_sizes):
            kernel = _pair(kernel_size)
            padding = tuple(value // 2 for value in kernel)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=padding))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(_pair(pool_size)))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        adaptive_size = _pair(config.get("adaptive_pool_output", 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(adaptive_size)
        flattened = in_channels * adaptive_size[0] * adaptive_size[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.get("classifier_dropout", 0.10))),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(inputs)))


class HuBERTMLP(nn.Module):
    """Configurable MLP for pooled HuBERT embeddings."""

    def __init__(self, input_dim: int, num_classes: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_dims = [int(value) for value in config.get("hidden_dims", [])]
        dropout = float(config.get("dropout", 0.0))
        use_batch_norm = bool(config.get("batch_norm", False))

        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            is_output = index == len(dims) - 2
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_output:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def build_model(config: Mapping[str, Any], num_classes: int) -> nn.Module:
    method = str(config["features"]["method"]).lower()
    if method == "mel":
        return MelCNN(num_classes, config["model"]["mel_cnn"])
    if method == "hubert":
        input_dim = int(config["features"]["hubert"]["embedding_dim"])
        return HuBERTMLP(input_dim, num_classes, config["model"]["hubert_mlp"])
    raise ValueError(f"Unsupported feature method: {method}")


def _expand(value: Any, length: int) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != length:
            raise ValueError(f"Expected {length} values, received {len(values)}.")
        return values
    return [value] * length


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(item) for item in value]
        if len(values) != 2:
            raise ValueError("A 2-D size must contain exactly two integers.")
        return values[0], values[1]
    size = int(value)
    return size, size
