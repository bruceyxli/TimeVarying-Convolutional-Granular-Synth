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
    _validate_ir_settings(sample_rate, target_ir_ms)
    if not isinstance(num_irs, (int, np.integer)) or not 1 <= num_irs <= 4096:
        raise ValueError("num_irs must be an integer between 1 and 4096")
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
        tau_ms = float(rng.uniform(min(6.0, target_ir_ms), target_ir_ms * 1.2))
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
    _validate_ir_settings(sample_rate, target_ir_ms)
    if max_files is not None and (not isinstance(max_files, int) or max_files < 1):
        raise ValueError("max_files must be a positive integer or None")
    exts = {'.wav', '.flac', '.aiff', '.aif', '.ogg'}
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if os.path.splitext(f)[1].lower() in exts]
    if max_files:
        files = files[:max_files]
    items: List[IRItem] = []
    for path in files:
        wav, sr = sf.read(path, dtype='float32', always_2d=False)
        if wav.size == 0 or not np.all(np.isfinite(wav)):
            raise ValueError(f"IR file is empty or contains non-finite samples: {path}")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        if sr != sample_rate:
            # Simple resampling
            gcd = np.gcd(sample_rate, sr)
            up = sample_rate // gcd
            down = sr // gcd
            wav = resample_poly(wav, up, down).astype(np.float32)
        target_len = max(8, int(sample_rate * target_ir_ms / 1000.0))
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


def _validate_ir_settings(sample_rate: int, target_ir_ms: float) -> None:
    if not isinstance(sample_rate, (int, np.integer)) or not 8000 <= sample_rate <= 192000:
        raise ValueError("sample_rate must be an integer between 8000 and 192000")
    if not np.isfinite(target_ir_ms) or not 0 < target_ir_ms <= 1000:
        raise ValueError("target_ir_ms must be between 0 (exclusive) and 1000")


class IRSelector:
    """Validate once and reuse selection data throughout a render."""

    STRATEGIES = frozenset(("fixed", "cycle", "random", "weighted", "centroid"))

    def __init__(self, ir_items: List[IRItem], strategy: str,
                 weights: Optional[np.ndarray] = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown IR strategy: {strategy}")
        if not ir_items:
            raise ValueError("A non-empty IR bank is required for wet standard rendering")
        self.items = []
        for item in ir_items:
            samples = np.asarray(item.samples, dtype=np.float32)
            if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
                raise ValueError("IR samples must be non-empty, finite, mono audio")
            if not np.isfinite(item.centroid_hz) or item.centroid_hz < 0:
                raise ValueError("IR centroids must be finite and non-negative")
            self.items.append(IRItem(samples, float(item.centroid_hz)))
        self.strategy = strategy
        self.count = len(self.items)
        self.counter = -1
        self.centroids = np.asarray([item.centroid_hz for item in self.items])
        self.weights = None
        if strategy == "weighted" and weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if (w.shape != (self.count,) or not np.all(np.isfinite(w))
                    or np.any(w < 0) or not np.any(w > 0)):
                raise ValueError("weights must match the IR bank, be finite, non-negative and have positive sum")
            # Scaling first prevents overflow with large but finite weights.
            w = w / np.max(w)
            self.weights = w / np.sum(w)

    def select(self, rng: np.random.Generator, grain_centroid_hz: Optional[float] = None) -> int:
        if self.strategy == "fixed":
            return 0
        if self.strategy == "cycle":
            self.counter = (self.counter + 1) % self.count
            return self.counter
        if self.strategy == "weighted" and self.weights is not None:
            return int(rng.choice(self.count, p=self.weights))
        if self.strategy == "centroid":
            if grain_centroid_hz is None or not np.isfinite(grain_centroid_hz):
                raise ValueError("centroid selection requires a finite grain centroid")
            diffs = np.abs(self.centroids - grain_centroid_hz)
            candidates = np.flatnonzero(diffs <= np.min(diffs) + 1e-6)
            return int(rng.choice(candidates))
        return int(rng.integers(0, self.count))


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
    selector = IRSelector(ir_items, strategy, weights)
    if strategy == "cycle" and cycle_counter:
        selector.counter = cycle_counter[0]
    result = selector.select(rng, grain_centroid_hz)
    if strategy == "cycle" and cycle_counter:
        cycle_counter[0] = selector.counter
    return result


