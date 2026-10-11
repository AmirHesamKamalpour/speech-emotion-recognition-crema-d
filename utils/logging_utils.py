from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def setup_logging(config_path: str | Path, log_file: str | Path) -> None:
    config_path = Path(config_path)
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logging.getLogger(__name__).warning(
            "Logging configuration not found at %s; using basicConfig.", config_path
        )
        return

    with config_path.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream)
    if "handlers" in config and "file" in config["handlers"]:
        config["handlers"]["file"]["filename"] = str(log_file)
    logging.config.dictConfig(config)
