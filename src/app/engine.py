"""Offline rendering; NumPy/SciPy operations are not audio-callback safe."""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, gcd
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import firwin, resample_poly, fftconvolve

from src.app.features import compute_spectral_centroid
from src.app.ir_bank import IRItem, IRSelector


@dataclass
class GranularConfig:
    sample_rate: int = 48000
    duration_sec: float = 10.0
    density_hz: float = 60.0
    grain_ms: float = 20.0
    jitter: float = 0.1
    pitch_semitones: float = 7.0
    pan_spread: float = 1.0
    wet: float = 0.6
    dry: float = 0.4
    normalize: bool = True
    ir_strategy: str = "weighted"
    rng_seed: Optional[int] = 2025
    variant: str = "standard"
    long_ir_ms: float = 120.0

    def validate(self) -> None:
        if (not isinstance(self.sample_rate, (int, np.integer))
                or not 8000 <= self.sample_rate <= 192000):
            raise ValueError("sample_rate must be an integer between 8000 and 192000")
        for name in ("duration_sec", "density_hz", "grain_ms", "long_ir_ms"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, maximum in (("jitter", 0.5), ("pan_spread", 1.0),
                              ("wet", 1.0), ("dry", 1.0), ("pitch_semitones", 24.0)):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
        if self.variant not in ("standard", "variant_a", "variant_b"):
            raise ValueError(f"Unknown variant: {self.variant}")
        if self.ir_strategy not in IRSelector.STRATEGIES:
            raise ValueError(f"Unknown IR strategy: {self.ir_strategy}")
        if self.rng_seed is not None and (
            not isinstance(self.rng_seed, (int, np.integer)) or self.rng_seed < 0
        ):
            raise ValueError("rng_seed must be a non-negative integer or None")
        # Offline limits are explicit, checked before allocating audio buffers.
        if not 1 <= self.sample_rate * self.duration_sec <= 115_200_000:
            raise ValueError("Output must contain 1 to 115200000 samples per channel")
        if self.duration_sec * self.density_hz > 1_000_000:
            raise ValueError("Render exceeds the 1000000 grain limit")
        if self.grain_ms > 1000 or self.long_ir_ms > 10000:
            raise ValueError("grain_ms must be <= 1000 and long_ir_ms <= 10000")


def _hann(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    return np.hanning(length).astype(np.float32)


def _equal_power_pan(pan: float) -> Tuple[float, float]:
    angle = (min(1.0, max(-1.0, pan)) + 1.0) * np.pi * 0.25
    return float(np.cos(angle)), float(np.sin(angle))


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=-1) if x.ndim > 1 else x


def _pitch_factors(semitone: float) -> Tuple[int, int]:
    # Preserve the original 1/100 pitch-ratio resolution.
    up, down = 100, max(1, int(round(100 * 2.0 ** (semitone / 12.0))))
    divisor = gcd(up, down)
    return up // divisor, down // divisor


@lru_cache(maxsize=384)
def _resample_filter(up: int, down: int) -> np.ndarray:
    """Cache SciPy's default Kaiser FIR, not audio data. Bounded across renders."""
    rate = max(up, down)
    taps = firwin(20 * rate + 1, 1.0 / rate, window=("kaiser", 5.0)).astype(np.float32)
    taps.flags.writeable = False
    return taps


def _pitch_resample(segment: np.ndarray, semitone: float, out_len: int) -> np.ndarray:
    up, down = _pitch_factors(semitone)
    # Raising pitch consumes MORE source samples. The old inverse ratio padded
    # most of an upward-shifted grain with silence (75% at +12 semitones).
    read_len = max(4, ceil(out_len * down / up))
    seg = np.asarray(segment[:read_len], dtype=np.float32)
    if len(seg) < read_len:
        seg = np.pad(seg, (0, read_len - len(seg)))
    if up == down:
        return seg[:out_len].copy()
    return resample_poly(seg, up, down, window=_resample_filter(up, down))[:out_len].astype(
        np.float32, copy=False
    )


def _render_grain(
    source: np.ndarray,
    start_idx: int,
    cfg: GranularConfig,
    rng: np.random.Generator,
    window: Optional[np.ndarray] = None,
    need_centroid: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float], float, np.ndarray]:
    """Return dry grain, centroid, pan gains, trigger jitter and source segment."""
    grain_len = max(8, int(cfg.sample_rate * cfg.grain_ms / 1000.0))
    semi = float(rng.uniform(-cfg.pitch_semitones, cfg.pitch_semitones))
    up, down = _pitch_factors(semi)
    read_len = max(4, ceil(grain_len * down / up))
    start = min(max(0, start_idx), max(0, len(source) - read_len))
    segment = source[start:start + read_len]
    dry = _pitch_resample(segment, semi, out_len=grain_len)
    dry *= _hann(grain_len) if window is None else window
    gains = _equal_power_pan(float(rng.uniform(-cfg.pan_spread, cfg.pan_spread)))
    centroid = compute_spectral_centroid(dry, cfg.sample_rate) if need_centroid else 0.0
    jitter = cfg.jitter / cfg.density_hz
    jitter_offset = float(rng.uniform(-jitter, jitter))
    return dry, centroid, gains, jitter_offset, segment


class _IRConvolver:
    """Lazy per-render IR spectra; discarded when the render ends."""

    def __init__(self, items: List[IRItem], grain_len: int):
        self.items = items
        self.grain_len = grain_len
        self.spectra = {}

    def convolve(self, grain: np.ndarray, index: int) -> np.ndarray:
        cached = self.spectra.get(index)
        if cached is None:
            samples = self.items[index].samples
            length = self.grain_len + len(samples) - 1
            size = next_fast_len(length, real=True)
            cached = (length, size, rfft(samples, n=size))
            self.spectra[index] = cached
        length, size, spectrum = cached
        transformed = rfft(grain, n=size)
        transformed *= spectrum
        return irfft(transformed, n=size)[:length]


def render_offline(
    source_audio: np.ndarray,
    ir_items: List[IRItem],
    cfg: GranularConfig,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    cfg.validate()
    source = np.asarray(source_audio, dtype=np.float32)
    if source.ndim not in (1, 2) or source.size == 0:
        raise ValueError("source_audio must be non-empty mono or frames-by-channels audio")
    if not np.all(np.isfinite(source)):
        raise ValueError("source_audio must contain only finite samples")
    src = _ensure_mono(source)
    sr = cfg.sample_rate
    total_len = int(sr * cfg.duration_sec)
    grain_len = max(8, int(sr * cfg.grain_ms / 1000.0))
    selector = None
    convolver = None
    if cfg.variant == "standard" and (cfg.wet > 0 or ir_items):
        selector = IRSelector(ir_items, cfg.ir_strategy, weights)
        if cfg.wet > 0:
            convolver = _IRConvolver(selector.items, grain_len)

    if cfg.variant == "variant_a":
        ir_len = max(8, int(sr * cfg.long_ir_ms / 1000.0))
        rngv = np.random.default_rng(cfg.rng_seed)
        long_ir = rngv.normal(0.0, 1.0, ir_len).astype(np.float32)
        long_ir *= np.exp(-np.linspace(0, 1.0, ir_len) * 6.0).astype(np.float32)
        long_ir /= np.max(np.abs(long_ir)) + 1e-12
        # Mix BEFORE granulation: dry=1 is the source, wet=1 the convolved source.
        # Keep the source scan length independent of Wet/Dry.
        if cfg.wet > 0:
            processed = fftconvolve(src, long_ir, mode="full")[:len(src)]
            src = cfg.dry * src + cfg.wet * processed
        else:
            src = cfg.dry * src

    # One output allocation; tails are clipped to the requested duration.
    mix = np.zeros((total_len, 2), dtype=np.float32)
    rng = np.random.default_rng(cfg.rng_seed)
    hop = 1.0 / cfg.density_hz
    num_grains = int(np.floor(cfg.duration_sec / hop))
    read_stride = max(1, len(src) // max(1, num_grains))
    window = _hann(grain_len)
    need_centroid = selector is not None and cfg.ir_strategy == "centroid"

    for i in range(num_grains):
        dry_grain, centroid, gains, jitter, segment = _render_grain(
            src, (i * read_stride) % len(src), cfg, rng, window, need_centroid
        )
        if cfg.variant == "variant_a":
            mono = dry_grain
        else:
            # Even at Wet=0 keep selection RNG draws, so changing Wet does not
            # change subsequent pitch/pan/jitter for a seeded render.
            index = selector.select(rng, centroid) if selector is not None else None
            if cfg.wet == 0:
                mono = dry_grain * cfg.dry
            else:
                if cfg.variant == "variant_b":
                    mono = fftconvolve(segment, dry_grain, mode="full").astype(np.float32)
                else:
                    mono = convolver.convolve(dry_grain, index)
                mono *= cfg.wet
                mono[:len(dry_grain)] += dry_grain * cfg.dry
        start = max(0, int(sr * (i * hop + jitter)))
        length = min(len(mono), total_len - start)
        if length > 0:
            mix[start:start + length, 0] += mono[:length] * gains[0]
            mix[start:start + length, 1] += mono[:length] * gains[1]

    if not np.all(np.isfinite(mix)):
        raise ValueError("Render overflowed; reduce input or IR amplitude")
    if cfg.normalize:
        peak = max(float(np.max(mix)), -float(np.min(mix)))
        if peak > 1.0:
            mix *= 0.99 / peak
    return mix


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, samplerate=sample_rate, subtype="PCM_24")
