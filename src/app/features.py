"""Audio feature extraction: spectral centroid and other utilities."""

import numpy as np


def compute_spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    """
    Compute spectral centroid (Hz) for a short-time frame.
    The input should be a short windowed segment. Returns a non-negative float.
    If the energy is extremely low, returns 0.
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=-1)
    x = signal.astype(np.float64)
    if not np.any(np.isfinite(x)):
        return 0.0
    x = x - np.mean(x)
    window = np.hanning(len(x)) if len(x) > 1 else np.ones_like(x)
    xw = x * window
    n_fft = int(1 << (len(xw) - 1).bit_length())
    spec = np.fft.rfft(xw, n=n_fft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    denom = np.sum(mag) + 1e-12
    centroid = float(np.sum(freqs * mag) / denom)
    return max(0.0, centroid)


