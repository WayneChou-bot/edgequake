"""PhaseNet picker backed by the original AI4EPS TensorFlow checkpoint.

Use when the SeisBench weight server is unreachable (weights live on GitHub):

    git clone --depth 1 https://github.com/AI4EPS/PhaseNet.git
    picker = TFPhaseNet(repo_dir="PhaseNet")

Checkpoint: model/190703-214543 (Zhu & Beroza 2019, trained on NCEDC).
Output channel order of the TF model is (noise, P, S) — already engine order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import Picker, PickerOutput


class TFPhaseNet(Picker):
    window_samples = 3000
    sampling_rate = 100.0
    name = "tf-phasenet-190703"

    def __init__(self, repo_dir: str = "PhaseNet"):
        repo = Path(repo_dir)
        ckpt_dir = repo / "model" / "190703-214543"
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"PhaseNet checkpoint not found at {ckpt_dir}; clone "
                "https://github.com/AI4EPS/PhaseNet.git first"
            )
        sys.path.insert(0, str(repo))
        import tensorflow as tf  # deferred import
        from phasenet.model import ModelConfig, UNet

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        config = ModelConfig(X_shape=[3000, 1, 3], Y_shape=[3000, 1, 3])
        self._model = UNet(config=config, mode="pred")
        sess_config = tf.compat.v1.ConfigProto()
        self._sess = tf.compat.v1.Session(config=sess_config)
        saver = tf.compat.v1.train.Saver()
        latest = tf.compat.v1.train.latest_checkpoint(str(ckpt_dir))
        saver.restore(self._sess, latest)

    def predict(self, window: np.ndarray) -> PickerOutput:
        x = self.normalize(window.astype(np.float32))  # (3, 3000)
        x = x.T[None, :, None, :]  # -> (1, 3000, 1, 3)
        preds = self._sess.run(
            self._model.preds,
            feed_dict={self._model.X: x, self._model.drop_rate: 0, self._model.is_training: False},
        )  # (1, 3000, 1, 3) = (noise, P, S)
        probs = preds[0, :, 0, :].T  # (3, 3000)
        return PickerOutput(probs=probs, sampling_rate=self.sampling_rate)
