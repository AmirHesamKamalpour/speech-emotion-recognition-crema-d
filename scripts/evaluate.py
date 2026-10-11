from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.metrics import classification_metrics
from utils.visualization import plot_confusion_matrix

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path or Path(config["training"]["checkpoint_dir"]) / "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = _load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for features, labels in tqdm(test_loader, desc="Test", leave=False):
            logits = model(features.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    label_names = list(config["dataset"]["label_order"])
    metrics = classification_metrics(targets, predictions, label_names)
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    _write_classification_report(metrics["classification_report"], output_dir / "classification_report.csv")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        label_names,
        output_dir / "confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        label_names,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    LOGGER.info("Final test accuracy: %.4f", metrics["accuracy"])
    return metrics


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location=device)


def _write_classification_report(report: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if isinstance(values, Mapping):
            rows.append({"label": label, **values})
        else:
            rows.append({"label": label, "accuracy": values})
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
