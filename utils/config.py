from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a required configuration value is missing or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and resolve project-relative paths."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    if not isinstance(config, dict):
        raise ConfigError("The root of config.yaml must be a mapping.")

    project_root = config_path.parent.parent
    config = deepcopy(config)
    config.setdefault("project", {})
    config["project"]["root"] = str(project_root)
    _resolve_paths(config, project_root)
    validate_config(config)
    return config


def _resolve_paths(config: dict[str, Any], project_root: Path) -> None:
    path_fields = {
        ("dataset", "root"),
        ("features", "cache", "directory"),
        ("training", "checkpoint_dir"),
        ("output", "directory"),
        ("logging", "config_file"),
    }
    for keys in path_fields:
        node: dict[str, Any] = config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                break
            node = node[key]
        else:
            final_key = keys[-1]
            value = node.get(final_key)
            if value:
                path = Path(value).expanduser()
                node[final_key] = str(path if path.is_absolute() else project_root / path)


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing required configuration value: {dotted_key}")
        value = value[key]
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    required = (
        "dataset.root",
        "dataset.emotion_codes",
        "dataset.label_order",
        "dataset.speakers",
        "audio.sample_rate",
        "audio.duration_seconds",
        "features.method",
        "split.train_size",
        "split.validation_size",
        "split.test_size",
        "training.batch_size",
        "training.epochs",
        "training.learning_rate",
        "training.optimizer.name",
        "training.loss.name",
    )
    for key in required:
        get_nested(config, key)

    method = str(get_nested(config, "features.method")).lower()
    if method not in {"mel", "hubert"}:
        raise ConfigError("features.method must be either 'mel' or 'hubert'.")

    fractions = [
        float(get_nested(config, "split.train_size")),
        float(get_nested(config, "split.validation_size")),
        float(get_nested(config, "split.test_size")),
    ]
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ConfigError("Split fractions must sum to 1.0.")

    if int(get_nested(config, "audio.sample_rate")) <= 0:
        raise ConfigError("audio.sample_rate must be positive.")
    if float(get_nested(config, "audio.duration_seconds")) <= 0:
        raise ConfigError("audio.duration_seconds must be positive.")


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False)
