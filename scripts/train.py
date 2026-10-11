from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.metrics import running_accuracy
from utils.visualization import plot_training_history

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, list[float]]:
    training_cfg = config["training"]
    epochs = int(training_cfg["epochs"])
    criterion = build_loss(training_cfg["loss"])
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg.get("scheduler", {}), epochs)
    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_accuracy = float("-inf")
    patience = int(training_cfg.get("early_stopping", {}).get("patience", 0))
    best_validation_loss = float("inf")
    min_delta = 1e-4
    epochs_without_improvement = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=training_cfg.get("gradient_clip_norm"),
            description=f"Train {epoch}/{epochs}",
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            description=f"Valid {epoch}/{epochs}",
        )
        history["train_loss"].append(train_result.loss)
        history["train_accuracy"].append(train_result.accuracy)
        history["validation_loss"].append(validation_result.loss)
        history["validation_accuracy"].append(validation_result.accuracy)

        _step_scheduler(scheduler, validation_result.loss)
        LOGGER.info(
            "Epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch,
            epochs,
            train_result.loss,
            train_result.accuracy,
            validation_result.loss,
            validation_result.accuracy,
        )

        state = _checkpoint_state(
            model, optimizer, scheduler, epoch, best_accuracy, history, config
        )
        torch.save(state, checkpoint_dir / "last.pt")

        improved = validation_result.loss < best_validation_loss - min_delta

        if improved:
            best_validation_loss = validation_result.loss
            epochs_without_improvement = 0

            state = _checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_accuracy=validation_result.accuracy,
                history=history,
                config=config,
            )
            state["best_validation_loss"] = best_validation_loss
            torch.save(state, checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            LOGGER.info("Early stopping triggered after %d unimproved epochs.", patience)
            break

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    plot_training_history(history, output_dir / "training_curves.png")
    return history


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    description: str,
    gradient_clip_norm: float | None = None,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for features, targets in tqdm(loader, desc=description, leave=False):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("DataLoader produced no examples.")
    return EpochResult(
        loss=total_loss / total_examples,
        accuracy=running_accuracy(total_correct, total_examples),
    )


def build_loss(config: Mapping[str, Any]) -> nn.Module:
    name = str(config["name"]).lower()
    if name not in {"cross_entropy", "crossentropy", "cross-entropy"}:
        raise ValueError("Only cross-entropy loss is supported for this classification task.")
    return nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )


def build_optimizer(model: nn.Module, training_cfg: Mapping[str, Any]) -> Optimizer:
    optimizer_cfg = training_cfg["optimizer"]
    name = str(optimizer_cfg["name"])
    optimizer_class = getattr(torch.optim, name, None)
    if optimizer_class is None:
        raise ValueError(f"Unknown torch optimizer: {name}")
    kwargs = {
        key: value
        for key, value in optimizer_cfg.items()
        if key != "name" and value is not None
    }
    kwargs["lr"] = float(training_cfg["learning_rate"])
    return optimizer_class(model.parameters(), **kwargs)


def build_scheduler(
    optimizer: Optimizer, config: Mapping[str, Any], epochs: int
) -> Any | None:
    if not config or not bool(config.get("enabled", False)):
        return None
    name = str(config["name"])
    kwargs = {key: value for key, value in config.items() if key not in {"enabled", "name"}}
    if name == "CosineAnnealingLR":
        kwargs.setdefault("T_max", epochs)
    scheduler_class = getattr(torch.optim.lr_scheduler, name, None)
    if scheduler_class is None:
        raise ValueError(f"Unknown learning-rate scheduler: {name}")
    return scheduler_class(optimizer, **kwargs)


def _step_scheduler(scheduler: Any | None, validation_loss: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def _checkpoint_state(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    epoch: int,
    best_accuracy: float,
    history: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_validation_accuracy": best_accuracy,
        "history": dict(history),
        "feature_method": config["features"]["method"],
        "label_order": list(config["dataset"]["label_order"]),
    }
