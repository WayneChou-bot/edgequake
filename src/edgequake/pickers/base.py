"""Picker interface — every phase picker plugs into the replay engine through this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PickerOutput:
    """Per-sample probabilities for one inference window.

    probs shape: (3, n_samples) — rows are (noise, P, S), values in [0, 1].
    """

    probs: np.ndarray
    sampling_rate: float

    @property
    def p_prob(self) -> np.ndarray:
        return self.probs[1]

    @property
    def s_prob(self) -> np.ndarray:
        return self.probs[2]


class Picker(ABC):
    """A phase picker maps a 3-component window to per-sample N/P/S probabilities."""

    #: window length (samples) the model expects; engine feeds exactly this many
    window_samples: int = 3000
    #: sampling rate the model expects (Hz); engine resamples input to match
    sampling_rate: float = 100.0
    name: str = "picker"

    @abstractmethod
    def predict(self, window: np.ndarray) -> PickerOutput:
        """window shape: (3, window_samples), channel order (Z, N, E) or (E, N, Z)
        — PhaseNet-family models are channel-order tolerant after normalization,
        but keep a consistent order per run."""

    @staticmethod
    def normalize(window: np.ndarray) -> np.ndarray:
        """Per-channel demean + std-normalize (PhaseNet convention)."""
        w = window - window.mean(axis=1, keepdims=True)
        std = w.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return w / std
