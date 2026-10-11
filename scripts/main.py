from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import (  # noqa: E402
    build_dataloaders,
    discover_records,
    precompute_datasets,
    save_split_manifest,
    stratified_split,
)
from models.model import build_model  # noqa: E402
from scripts.evaluate import evaluate_model  # noqa: E402
from scripts.train import train_model  # noqa: E402
from utils.config import load_config, save_yaml  # noqa: E402
from utils.features import build_feature_extractor  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from utils.reproducibility import seed_everything  # noqa: E402

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CREMA-D speech emotion recognition")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "all"),
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used in evaluate mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        config["logging"]["config_file"],
        output_dir / config["logging"].get("log_filename", "experiment.log"),
    )

    seed_everything(
        int(config["reproducibility"]["seed"]),
        deterministic=bool(config["reproducibility"].get("deterministic", True)),
    )
    save_yaml(config, output_dir / "resolved_config.yaml")
    device = resolve_device(str(config["training"].get("device", "auto")))
    LOGGER.info("Training device: %s", device)
    LOGGER.info("Feature method: %s", config["features"]["method"])

    feature_extractor = build_feature_extractor(config)
    records = discover_records(config)
    splits = stratified_split(records, config)
    save_split_manifest(splits, output_dir / "split_manifest.csv")
    train_loader, validation_loader, test_loader = build_dataloaders(
        splits, config, feature_extractor
    )

    if bool(config["features"]["cache"].get("precompute", False)):
        precompute_datasets((train_loader, validation_loader, test_loader))

    num_classes = len(config["dataset"]["label_order"])
    model = build_model(config, num_classes)
    LOGGER.info("Classifier parameters: %d", sum(p.numel() for p in model.parameters()))

    if args.mode in {"train", "all"}:
        train_model(model, train_loader, validation_loader, config, device)
    if args.mode in {"evaluate", "all"}:
        metrics = evaluate_model(model, test_loader, config, device, args.checkpoint)
        print(f"Test accuracy: {metrics['accuracy']:.4f}")


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


if __name__ == "__main__":
    main()
