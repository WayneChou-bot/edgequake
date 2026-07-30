"""Fixed-capacity ring buffer for continuous 3-component waveform streaming."""
from __future__ import annotations

import numpy as np


class RingBuffer:
    """Holds the most recent `capacity` samples of a 3-channel stream.

    Windows are read out as contiguous arrays (unrolled), oldest -> newest.
    Before the buffer has filled once, the missing head is zero-padded so the
    consumer always receives a full-length window (mirrors cold-start of a
    real-time system).
    """

    def __init__(self, capacity: int, n_channels: int = 3, dtype=np.float32):
        self.capacity = int(capacity)
        self._buf = np.zeros((n_channels, self.capacity), dtype=dtype)
        self._write = 0          # next write position
        self._count = 0          # total samples ever written (saturates at capacity)

    @property
    def filled(self) -> bool:
        return self._count >= self.capacity

    def push(self, chunk: np.ndarray) -> None:
        """chunk shape: (n_channels, n) with n <= capacity."""
        n = chunk.shape[1]
        if n == 0:
            return
        if n >= self.capacity:  # keep only the tail
            self._buf[:] = chunk[:, -self.capacity:]
            self._write = 0
            self._count = self.capacity
            return
        end = self._write + n
        if end <= self.capacity:
            self._buf[:, self._write:end] = chunk
        else:
            k = self.capacity - self._write
            self._buf[:, self._write:] = chunk[:, :k]
            self._buf[:, : end - self.capacity] = chunk[:, k:]
        self._write = end % self.capacity
        self._count = min(self._count + n, self.capacity)

    def window(self) -> np.ndarray:
        """Return the latest `capacity` samples, oldest first (zero-padded head
        while cold-starting)."""
        out = np.concatenate(
            [self._buf[:, self._write:], self._buf[:, : self._write]], axis=1
        )
        if not self.filled:
            pad = self.capacity - self._count
            out = np.concatenate(
                [np.zeros((out.shape[0], pad), dtype=out.dtype), out[:, pad:]], axis=1
            ) if pad else out
            # zero the not-yet-written head explicitly
            out[:, : self.capacity - self._count] = 0.0
        return out
