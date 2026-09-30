"""Audio feature extraction: spectral centroid and other utilities."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _centroid_basis(length: int, sample_rate: int):
    window = np.hanning(length) if length > 1 else np.ones(length)
    n_fft = 1 << (length - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    window.flags.writeable = False
    freqs.flags.writeable = False
    return window, n_fft, freqs


def compute_spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    """
    Compute spectral centroid (Hz) for a short-time frame.
    The input should be a short windowed segment. Returns a non-negative float.
    Empty, non-finite and silent frames return 0.
    """
    signal = np.asarray(signal)
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        return 0.0
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=-1)
    x = signal.astype(np.float64)
    x = x - np.mean(x)
    window, n_fft, freqs = _centroid_basis(len(x), sample_rate)
    xw = x * window
    spec = np.fft.rfft(xw, n=n_fft)
    mag = np.abs(spec)
    denom = np.sum(mag) + 1e-12
    centroid = float(np.sum(freqs * mag) / denom)
    return max(0.0, centroid)


