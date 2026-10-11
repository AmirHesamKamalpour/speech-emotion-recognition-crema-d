from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn


class AudioFeatureExtractor(ABC):
    """Interface shared by all audio feature extractors."""

    method: str

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """Feature shape excluding the batch dimension."""

    @abstractmethod
    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Extract one feature tensor from a mono waveform."""


class LogMelFeatureExtractor(AudioFeatureExtractor):
    method = "mel"

    def __init__(self, config: Mapping[str, Any]) -> None:
        audio_cfg = config["audio"]
        mel_cfg = config["features"]["mel"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.duration_seconds = float(audio_cfg["duration_seconds"])
        self.n_mels = int(mel_cfg["n_mels"])
        self.standardize = bool(mel_cfg.get("standardize", False))
        self.epsilon = float(mel_cfg.get("standardize_epsilon", 1e-6))

        f_max = mel_cfg.get("f_max")
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(mel_cfg["n_fft"]),
            win_length=_optional_int(mel_cfg.get("win_length")),
            hop_length=int(mel_cfg["hop_length"]),
            f_min=float(mel_cfg.get("f_min", 0.0)),
            f_max=None if f_max is None else float(f_max),
            pad=int(mel_cfg.get("pad", 0)),
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            power=float(mel_cfg.get("power", 2.0)),
            normalized=bool(mel_cfg.get("normalized", False)),
            center=bool(mel_cfg.get("center", True)),
            pad_mode=str(mel_cfg.get("pad_mode", "reflect")),
            norm=mel_cfg.get("mel_norm"),
            mel_scale=str(mel_cfg.get("mel_scale", "htk")),
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=mel_cfg.get("top_db"),
        )

        target_samples = round(self.sample_rate * self.duration_seconds)
        dummy = torch.zeros(1, target_samples)
        with torch.no_grad():
            shape = self.to_db(self.mel_transform(dummy)).shape
        self._output_shape = tuple(int(value) for value in shape)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        _validate_sample_rate(sample_rate, self.sample_rate)
        waveform = _as_mono_2d(waveform)
        features = self.to_db(self.mel_transform(waveform))
        if self.standardize:
            mean = features.mean()
            std = features.std(unbiased=False)
            features = (features - mean) / std.clamp_min(self.epsilon)
        return features.to(dtype=torch.float32)


class HuBERTFeatureExtractor(AudioFeatureExtractor):
    """Frozen TorchAudio HuBERT feature extractor with configurable pooling."""

    method = "hubert"

    def __init__(self, config: Mapping[str, Any]) -> None:
        audio_cfg = config["audio"]
        hubert_cfg = config["features"]["hubert"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.target_samples = round(
            self.sample_rate * float(audio_cfg["duration_seconds"])
        )
        self.embedding_dim = int(hubert_cfg["embedding_dim"])
        self.pooling = str(hubert_cfg.get("pooling", "mean")).lower()
        self.device = _resolve_device(str(hubert_cfg.get("device", "auto")))

        bundle_name = str(hubert_cfg["torchaudio_bundle"])
        if not hasattr(torchaudio.pipelines, bundle_name):
            available = sorted(
                name for name in dir(torchaudio.pipelines) if name.startswith("HUBERT")
            )
            raise ValueError(
                f"Unknown TorchAudio HuBERT bundle '{bundle_name}'. Available: {available}"
            )
        bundle = getattr(torchaudio.pipelines, bundle_name)
        if int(bundle.sample_rate) != self.sample_rate:
            raise ValueError(
                f"{bundle_name} expects {bundle.sample_rate} Hz, but audio.sample_rate "
                f"is {self.sample_rate} Hz."
            )

        try:
            from transformers import Wav2Vec2FeatureExtractor
        except ImportError as error:
            raise ImportError(
                "HuBERT feature extraction requires the transformers package. "
                "Install the project requirements before using features.method=hubert."
            ) from error

        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=int(hubert_cfg.get("feature_size", 1)),
            sampling_rate=self.sample_rate,
            padding_value=float(hubert_cfg.get("padding_value", 0.0)),
            do_normalize=bool(hubert_cfg.get("do_normalize", True)),
            return_attention_mask=True,
        )
        self.model: nn.Module = bundle.get_model().to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.embedding_dim,)

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        _validate_sample_rate(sample_rate, self.sample_rate)
        mono = _as_mono_2d(waveform).squeeze(0).detach().cpu().numpy()
        processed = self.processor(
            mono,
            sampling_rate=sample_rate,
            padding="max_length",
            max_length=self.target_samples,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = processed.input_values.to(self.device)
        attention_mask = processed.attention_mask.to(self.device)
        input_lengths = attention_mask.sum(dim=-1).to(dtype=torch.long)

        with torch.inference_mode():
            hidden_layers, output_lengths = self.model.extract_features(
                input_values, input_lengths
            )
            last_hidden_state = hidden_layers[-1]
            pooled = self._pool(last_hidden_state, output_lengths)

        if pooled.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"Configured HuBERT embedding_dim={self.embedding_dim}, but the model "
                f"returned {pooled.shape[-1]}."
            )
        return pooled.squeeze(0).to(device="cpu", dtype=torch.float32)

    def _pool(self, hidden: Tensor, lengths: Tensor | None) -> Tensor:
        if lengths is None:
            lengths = torch.full(
                (hidden.shape[0],), hidden.shape[1], device=hidden.device, dtype=torch.long
            )
        lengths = lengths.to(device=hidden.device).clamp(min=1, max=hidden.shape[1])
        steps = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)

        if self.pooling == "mean":
            masked = hidden * mask.unsqueeze(-1)
            return masked.sum(dim=1) / lengths.unsqueeze(-1)
        if self.pooling == "max":
            masked = hidden.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return masked.max(dim=1).values
        if self.pooling == "first":
            return hidden[:, 0, :]
        raise ValueError("HuBERT pooling must be one of: mean, max, first.")


def build_feature_extractor(config: Mapping[str, Any]) -> AudioFeatureExtractor:
    method = str(config["features"]["method"]).lower()
    if method == "mel":
        return LogMelFeatureExtractor(config)
    if method == "hubert":
        return HuBERTFeatureExtractor(config)
    raise ValueError(f"Unsupported feature method: {method}")


def _as_mono_2d(waveform: Tensor) -> Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {waveform.shape}.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(dtype=torch.float32)


def _validate_sample_rate(actual: int, expected: int) -> None:
    if int(actual) != int(expected):
        raise ValueError(f"Expected {expected} Hz audio after preprocessing, got {actual} Hz.")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for HuBERT extraction but is unavailable.")
    return device
