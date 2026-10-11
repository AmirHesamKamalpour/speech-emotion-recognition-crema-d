from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
import torchaudio
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.features import AudioFeatureExtractor
from utils.reproducibility import make_generator, seed_worker

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioRecord:
    path: str
    speaker_id: int
    emotion_code: str
    emotion: str
    label: int


@dataclass(frozen=True)
class DatasetSplits:
    train: list[AudioRecord]
    validation: list[AudioRecord]
    test: list[AudioRecord]


class CREMADataset(Dataset[tuple[Tensor, Tensor]]):
    """CREMA-D dataset with configurable audio preprocessing and feature caching."""

    def __init__(
        self,
        records: Sequence[AudioRecord],
        config: Mapping[str, Any],
        feature_extractor: AudioFeatureExtractor,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.feature_extractor = feature_extractor
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.target_samples = round(
            self.sample_rate * float(config["audio"]["duration_seconds"])
        )
        self.pad_mode = str(config["audio"].get("pad_mode", "constant"))
        self.crop_mode = str(config["audio"].get("crop_mode", "center"))
        self.cache_enabled = bool(config["features"]["cache"].get("enabled", False))
        self.cache_dir = Path(config["features"]["cache"]["directory"]) / feature_extractor.method
        self.cache_signature = _feature_signature(config)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        cache_path = self._cache_path(record)
        if cache_path is not None and cache_path.is_file():
            feature = _safe_torch_load(cache_path)
        else:
            waveform, original_rate = torchaudio.load(record.path)
            waveform = _to_mono(waveform)
            waveform = _resample(waveform, original_rate, self.sample_rate)
            waveform = _fix_duration(
                waveform,
                self.target_samples,
                crop_mode=self.crop_mode,
                pad_mode=self.pad_mode,
            )
            feature = self.feature_extractor(waveform, self.sample_rate)
            if cache_path is not None:
                _atomic_torch_save(feature, cache_path)

        return feature, torch.tensor(record.label, dtype=torch.long)

    def _cache_path(self, record: AudioRecord) -> Path | None:
        if not self.cache_enabled:
            return None
        file_identity = hashlib.sha256(str(Path(record.path).resolve()).encode()).hexdigest()[:16]
        name = f"{Path(record.path).stem}-{file_identity}-{self.cache_signature}.pt"
        return self.cache_dir / name


def discover_records(config: Mapping[str, Any]) -> list[AudioRecord]:
    dataset_cfg = config["dataset"]
    root = Path(dataset_cfg["root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"CREMA-D directory not found: {root}. Put WAV files there or update dataset.root."
        )

    extension = str(dataset_cfg.get("extension", ".wav"))
    separator = str(dataset_cfg["filename"]["separator"])
    speaker_index = int(dataset_cfg["filename"]["speaker_index"])
    emotion_index = int(dataset_cfg["filename"]["emotion_index"])
    recursive = bool(dataset_cfg.get("recursive", False))
    emotion_codes = {str(k): str(v) for k, v in dataset_cfg["emotion_codes"].items()}
    label_order = [str(label) for label in dataset_cfg["label_order"]]
    label_to_index = {label: index for index, label in enumerate(label_order)}
    selected_speakers = _speaker_ids(dataset_cfg["speakers"])

    pattern = f"*{extension}"
    paths = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    records: list[AudioRecord] = []
    malformed: list[str] = []

    for path in paths:
        tokens = path.stem.split(separator)
        try:
            speaker_id = int(tokens[speaker_index])
            emotion_code = tokens[emotion_index]
        except (IndexError, ValueError):
            malformed.append(path.name)
            continue
        if speaker_id not in selected_speakers or emotion_code not in emotion_codes:
            continue
        emotion = emotion_codes[emotion_code]
        if emotion not in label_to_index:
            raise ValueError(
                f"Emotion '{emotion}' from code '{emotion_code}' is missing from dataset.label_order."
            )
        records.append(
            AudioRecord(
                path=str(path),
                speaker_id=speaker_id,
                emotion_code=emotion_code,
                emotion=emotion,
                label=label_to_index[emotion],
            )
        )

    if malformed:
        LOGGER.warning("Skipped %d malformed filenames; first examples: %s", len(malformed), malformed[:5])
    if not records:
        raise RuntimeError(
            "No matching CREMA-D WAV files were found. Check dataset.root, speaker IDs, "
            "emotion codes, extension and filename indices."
        )

    _validate_class_counts(records, label_order)
    LOGGER.info("Discovered %d matching clips from %d speakers.", len(records), len(selected_speakers))
    return records


def stratified_split(records: Sequence[AudioRecord], config: Mapping[str, Any]) -> DatasetSplits:
    split_cfg = config["split"]
    seed = int(config["reproducibility"]["seed"])
    train_size = float(split_cfg["train_size"])
    validation_size = float(split_cfg["validation_size"])
    test_size = float(split_cfg["test_size"])
    labels = [record.label for record in records]

    train_records, temporary_records = train_test_split(
        list(records),
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    validation_fraction_of_temp = validation_size / (validation_size + test_size)
    temporary_labels = [record.label for record in temporary_records]
    validation_records, test_records = train_test_split(
        temporary_records,
        train_size=validation_fraction_of_temp,
        random_state=seed,
        shuffle=True,
        stratify=temporary_labels,
    )
    return DatasetSplits(train_records, validation_records, test_records)


def build_dataloaders(
    splits: DatasetSplits,
    config: Mapping[str, Any],
    feature_extractor: AudioFeatureExtractor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    training_cfg = config["training"]
    seed = int(config["reproducibility"]["seed"])
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(training_cfg.get("persistent_workers", False)) and num_workers > 0

    if feature_extractor.method == "hubert" and num_workers > 0:
        raise ValueError(
            "HuBERT extraction owns a pretrained model and must use training.num_workers=0. "
            "Enable feature caching/precomputation to keep later epochs fast."
        )

    datasets = [
        CREMADataset(splits.train, config, feature_extractor),
        CREMADataset(splits.validation, config, feature_extractor),
        CREMADataset(splits.test, config, feature_extractor),
    ]
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    train_loader = DataLoader(
        datasets[0],
        shuffle=True,
        generator=make_generator(seed),
        drop_last=bool(training_cfg.get("drop_last", False)),
        **common,
    )
    validation_loader = DataLoader(datasets[1], shuffle=False, **common)
    test_loader = DataLoader(datasets[2], shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def precompute_datasets(loaders: Iterable[DataLoader]) -> None:
    """Populate configured feature caches before model training."""
    for loader in loaders:
        dataset = loader.dataset
        if not isinstance(dataset, CREMADataset) or not dataset.cache_enabled:
            continue
        description = f"Caching {dataset.feature_extractor.method} features"
        for index in tqdm(range(len(dataset)), desc=description):
            dataset[index]


def save_split_manifest(splits: DatasetSplits, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["split", "path", "speaker_id", "emotion_code", "emotion", "label"],
        )
        writer.writeheader()
        for split_name, records in asdict(splits).items():
            for record in records:
                writer.writerow({"split": split_name, **record})


def _speaker_ids(config: Mapping[str, Any]) -> set[int]:
    explicit = config.get("ids")
    if explicit:
        return {int(value) for value in explicit}
    start = int(config["start"])
    end = int(config["end"])
    if end < start:
        raise ValueError("dataset.speakers.end must be greater than or equal to start.")
    return set(range(start, end + 1))


def _validate_class_counts(records: Sequence[AudioRecord], label_order: Sequence[str]) -> None:
    counts = {label: 0 for label in label_order}
    for record in records:
        counts[record.emotion] += 1
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"No clips found for configured emotions: {missing}")
    LOGGER.info("Class counts: %s", counts)


def _to_mono(waveform: Tensor) -> Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"torchaudio.load returned unexpected shape: {waveform.shape}")
    return waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform


def _resample(waveform: Tensor, source_rate: int, target_rate: int) -> Tensor:
    if int(source_rate) == int(target_rate):
        return waveform
    return torchaudio.functional.resample(waveform, int(source_rate), int(target_rate))


def _fix_duration(waveform: Tensor, target_samples: int, crop_mode: str, pad_mode: str) -> Tensor:
    current = waveform.shape[-1]
    if current > target_samples:
        if crop_mode == "center":
            start = (current - target_samples) // 2
        elif crop_mode == "start":
            start = 0
        elif crop_mode == "random":
            start = int(torch.randint(0, current - target_samples + 1, (1,)).item())
        else:
            raise ValueError("audio.crop_mode must be center, start, or random.")
        return waveform[..., start : start + target_samples]
    if current < target_samples:
        if pad_mode != "constant":
            raise ValueError("Only constant zero-padding is supported for 1-D audio clips.")
        return F.pad(waveform, (0, target_samples - current), mode="constant", value=0.0)
    return waveform


def _feature_signature(config: Mapping[str, Any]) -> str:
    relevant = {
        "audio": config["audio"],
        "features": {
            "method": config["features"]["method"],
            str(config["features"]["method"]): config["features"][str(config["features"]["method"])],
        },
    }
    payload = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_torch_load(path: Path) -> Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Tensor):
        raise TypeError(f"Cached feature is not a tensor: {path}")
    return value


def _atomic_torch_save(tensor: Tensor, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(tensor.detach().cpu(), temporary)
    os.replace(temporary, path)
