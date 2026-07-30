"""SeisBench-backed picker — the default path on a normal laptop.

Requires internet access to the SeisBench model repository on first use
(weights are cached under ~/.seisbench afterwards).
"""
from __future__ import annotations

import numpy as np

from .base import Picker, PickerOutput


class SeisBenchPhaseNet(Picker):
    """Pretrained PhaseNet via SeisBench (`from_pretrained`).

    weights: "original" (NCEDC, Zhu & Beroza 2019), "stead", "instance", ...
    See https://seisbench.readthedocs.io for the full list.
    """

    window_samples = 3001
    sampling_rate = 100.0

    def __init__(self, weights: str = "original", state_dict_path: str | None = None,
                 labels_override: str | None = None):
        """state_dict_path: optional .pt file (e.g. a fine-tuned checkpoint from
        the Kaggle notebook). The base `weights` are loaded first to fix the
        architecture, then the state dict overrides the parameters."""
        import seisbench.models as sbm  # deferred import

        self.model = sbm.PhaseNet.from_pretrained(weights)
        if state_dict_path:
            import torch

            ckpt = torch.load(state_dict_path, map_location="cpu",
                              weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(sd)
            from pathlib import Path

            self.name = f"seisbench-phasenet-ft-{Path(state_dict_path).stem}"
        else:
            self.name = f"seisbench-phasenet-{weights}"
        # labels_override: reinterpret output channel order, e.g. "PSN" for a
        # checkpoint fine-tuned against mismatched labeller channel order
        self._labels = labels_override or getattr(self.model, "labels", "NPS")
        if labels_override:
            self.name += f"-labels{labels_override}"
        self.model.eval()

    def predict(self, window: np.ndarray) -> PickerOutput:
        import torch

        x = self.normalize(window.astype(np.float32))
        with torch.no_grad():
            y = self.model(torch.from_numpy(x[None]))  # (1, 3, N)
        probs = y[0].numpy()
        # reorder model output channels to engine convention (noise, P, S)
        idx = {c: i for i, c in enumerate(self._labels)}
        probs = np.stack([probs[idx["N"]], probs[idx["P"]], probs[idx["S"]]])
        return PickerOutput(probs=probs, sampling_rate=self.sampling_rate)
