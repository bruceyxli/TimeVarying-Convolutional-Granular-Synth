import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly

from src.app.features import compute_spectral_centroid


@dataclass
class IRItem:
    samples: np.ndarray  # mono
    centroid_hz: float


def _butter_bandpass(low_hz: float, high_hz: float, fs: int, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    low = max(1.0, low_hz) / (fs * 0.5)
    high = min(fs * 0.5 - 100.0, high_hz) / (fs * 0.5)
    low = max(1e-6, min(0.99, low))
    high = max(low + 1e-6, min(0.999, high))
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _exp_decay(length: int, sr: int, tau_ms: float) -> np.ndarray:
    t = np.arange(length) / sr
    tau = max(1e-3, tau_ms / 1000.0)
    return np.exp(-t / tau)


def _normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)


def _align_peak_to_zero(ir: np.ndarray) -> np.ndarray:
    # Align the largest peak to the start so that it acts as t=0
    idx = int(np.argmax(np.abs(ir)))
    return np.roll(ir, -idx)


def generate_demo_ir_bank(
    num_irs: int = 64,
    target_ir_ms: float = 16.0,
    sample_rate: int = 48000,
    seed: Optional[int] = 2025,
) -> List[IRItem]:
    """
    Generate a bank of short IRs: white-noise impulse -> random bandpass -> exponential decay,
    then align the peak to t=0 and normalize.
    """
    rng = np.random.default_rng(seed)
    ir_len = int(sample_rate * target_ir_ms / 1000.0)
    ir_len = max(8, ir_len)
    items: List[IRItem] = []

    for _ in range(num_irs):
        base = rng.normal(0.0, 1.0, ir_len).astype(np.float32)
        # Randomly choose bandpass range
        low = float(rng.choice([80, 150, 300, 600, 1200]))
        high = low * float(rng.uniform(2.0, 5.0))
        b, a = _butter_bandpass(low, high, sample_rate, order=2)
        colored = lfilter(b, a, base)
        # Exponential decay with random time constant
        tau_ms = float(rng.uniform(6.0, target_ir_ms * 1.2))
        env = _exp_decay(ir_len, sample_rate, tau_ms=tau_ms)
        ir = colored * env
        ir = _align_peak_to_zero(ir)
        ir = _normalize_peak(ir).astype(np.float32)
        centroid = compute_spectral_centroid(ir, sample_rate)
        items.append(IRItem(samples=ir, centroid_hz=centroid))
    return items


def load_ir_folder(
    folder: str,
    sample_rate: int = 48000,
    target_ir_ms: float = 16.0,
    max_files: Optional[int] = None,
) -> List[IRItem]:
    """
    Load IRs from a folder (wav/flac/etc.). If longer than target length, take a segment
    around the maximum peak; if sample rate differs, resample.
    """
    exts = {'.wav', '.flac', '.aiff', '.aif', '.ogg'}
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in exts]
    if max_files:
        files = files[:max_files]
    items: List[IRItem] = []
    for path in files:
        wav, sr = sf.read(path, dtype='float32', always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        if sr != sample_rate:
            # Simple resampling
            gcd = np.gcd(sample_rate, sr)
            up = sample_rate // gcd
            down = sr // gcd
            wav = resample_poly(wav, up, down).astype(np.float32)
        target_len = int(sample_rate * target_ir_ms / 1000.0)
        if len(wav) >= target_len:
            # take a segment near the maximum peak
            idx = int(np.argmax(np.abs(wav)))
            start = max(0, idx - target_len // 8)
            seg = wav[start:start + target_len]
            if len(seg) < target_len:
                seg = np.pad(seg, (0, target_len - len(seg)))
        else:
            seg = np.pad(wav, (0, target_len - len(wav)))
        seg = _align_peak_to_zero(seg)
        seg = _normalize_peak(seg).astype(np.float32)
        centroid = compute_spectral_centroid(seg, sample_rate)
        items.append(IRItem(samples=seg, centroid_hz=centroid))
    return items


def select_ir_index(
    strategy: str,
    rng: np.random.Generator,
    ir_items: List[IRItem],
    grain_centroid_hz: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    cycle_counter: Optional[List[int]] = None,
) -> int:
    """
    IR selection strategies:
    - fixed: always 0
    - cycle: round-robin (requires external counter list with one integer)
    - random: uniform random
    - weighted: random with given weights (same length as bank)
    - centroid: pick IR whose centroid is closest to grain's centroid (ties broken randomly)
    """
    n = len(ir_items)
    if n == 0:
        return 0
    st = (strategy or 'random').lower()
    if st == 'fixed':
        return 0
    if st == 'cycle':
        if cycle_counter is None or not cycle_counter:
            return 0
        cycle_counter[0] = (cycle_counter[0] + 1) % n
        return cycle_counter[0]
    if st == 'weighted' and weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != n or np.all(w <= 0):
            return int(rng.integers(0, n))
        w = w / np.sum(w)
        return int(rng.choice(np.arange(n), p=w))
    if st == 'centroid' and grain_centroid_hz is not None:
        cents = np.array([it.centroid_hz for it in ir_items], dtype=np.float64)
        diffs = np.abs(cents - float(grain_centroid_hz))
        min_diff = np.min(diffs)
        cand = np.where(diffs <= (min_diff + 1e-6))[0]
        return int(rng.choice(cand))
    # default: random
    return int(rng.integers(0, n))


