from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], label_names: Sequence[str]
) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="weighted", zero_division=0
    )[2]
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=list(label_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }


def running_accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0
