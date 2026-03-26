from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve

from src.app.features import compute_spectral_centroid
from src.app.ir_bank import IRItem, select_ir_index


@dataclass
class GranularConfig:
    sample_rate: int = 48000
    duration_sec: float = 10.0
    density_hz: float = 60.0              # grains per second
    grain_ms: float = 20.0                # grain output length (ms)
    jitter: float = 0.1                   # trigger time jitter ratio (0~0.5)
    pitch_semitones: float = 7.0          # pitch random range (±semitones)
    pan_spread: float = 1.0               # stereo pan spread (0 center, 1 full)
    wet: float = 0.6                      # wet ratio
    dry: float = 0.4                      # dry ratio
    normalize: bool = True                # normalize after rendering
    ir_strategy: str = "weighted"         # 'fixed'/'cycle'/'random'/'weighted'/'centroid'
    rng_seed: Optional[int] = 2025
    # variants
    variant: str = "standard"             # 'standard' | 'variant_a' | 'variant_b'
    long_ir_ms: float = 120.0             # used by variant_a


def _hann(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    return np.hanning(length).astype(np.float32)


def _equal_power_pan(pan: float) -> Tuple[float, float]:
    """
    pan in [-1,1] -> (gainL, gainR), equal-power panning
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    x = (pan + 1.0) * 0.5
    left = np.cos(x * np.pi * 0.5)
    right = np.sin(x * np.pi * 0.5)
    return float(left), float(right)


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=-1) if x.ndim > 1 else x


def _pitch_resample(segment: np.ndarray, semitone: float, out_len: int) -> np.ndarray:
    """
    Change pitch by resampling: ratio = 2^(semitone/12), and then resample to fixed out_len.
    """
    ratio = 2.0 ** (semitone / 12.0)
    # To keep output length constant: read roughly out_len/ratio, then resample to out_len
    read_len = max(4, int(out_len / ratio))
    seg = segment[:read_len]
    if len(seg) < read_len:
        seg = np.pad(seg, (0, read_len - len(seg)))
    ratio_inv = 1.0 / ratio
    up = 100
    down = max(1, int(round(up / ratio_inv)))
    pitched = resample_poly(seg, up, down).astype(np.float32)
    if len(pitched) < out_len:
        pitched = np.pad(pitched, (0, out_len - len(pitched)))
    else:
        pitched = pitched[:out_len]
    return pitched


def _render_grain(
    source: np.ndarray,
    start_idx: int,
    cfg: GranularConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Tuple[float, float], float, np.ndarray]:
    """
    Returns: dry_grain(mono), grain_centroid_hz, (gainL, gainR), jittered_start_time_sec
    """
    sr = cfg.sample_rate
    grain_len = int(sr * cfg.grain_ms / 1000.0)
    grain_len = max(8, grain_len)
    semi = float(rng.uniform(-cfg.pitch_semitones, cfg.pitch_semitones))
    read_len = max(4, int(grain_len / (2.0 ** (semi / 12.0))))
    start = int(np.clip(start_idx, 0, max(0, len(source) - read_len)))
    segment = source[start:start + read_len].astype(np.float32)
    pitched = _pitch_resample(segment, semi, out_len=grain_len)
    window = _hann(grain_len)
    dry = pitched * window
    pan = float(rng.uniform(-cfg.pan_spread, cfg.pan_spread))
    gain_l, gain_r = _equal_power_pan(pan)
    centroid = compute_spectral_centroid(dry, sr)
    base_hop = 1.0 / max(1e-6, cfg.density_hz)
    jitter = float(cfg.jitter) * base_hop
    jitter_offset = float(rng.uniform(-jitter, jitter))
    return dry, centroid, (gain_l, gain_r), jitter_offset, segment


def render_offline(
    source_audio: np.ndarray,
    ir_items: List[IRItem],
    cfg: GranularConfig,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    sr = cfg.sample_rate
    src = _ensure_mono(source_audio.astype(np.float32))
    # Variant A: pre-convolution with a longer IR, then granular without per-grain convolution
    if cfg.variant == "variant_a":
        ir_len = int(sr * max(8.0, cfg.long_ir_ms) / 1000.0)
        rngv = np.random.default_rng(cfg.rng_seed)
        base = rngv.normal(0.0, 1.0, ir_len).astype(np.float32)
        decay = np.exp(-np.linspace(0, 1.0, ir_len) * 6.0).astype(np.float32)
        long_ir = base * decay
        long_ir = long_ir / (np.max(np.abs(long_ir)) + 1e-12)
        src = fftconvolve(src, long_ir, mode='full').astype(np.float32)
    total_len = int(sr * cfg.duration_sec)
    out_l = np.zeros(total_len + 48000, dtype=np.float32)
    out_r = np.zeros_like(out_l)
    dry_l = np.zeros_like(out_l)
    dry_r = np.zeros_like(out_l)

    rng = np.random.default_rng(cfg.rng_seed)
    grain_len = int(sr * cfg.grain_ms / 1000.0)
    hop = max(1e-6, 1.0 / cfg.density_hz)
    num_grains = int(np.floor(cfg.duration_sec / hop))
    cycle_counter = [ -1 ]

    if len(src) < 8:
        src = rng.normal(0.0, 0.2, int(sr * cfg.duration_sec)).astype(np.float32)
    read_stride = max(1, int(len(src) / max(1, num_grains)))

    for i in range(num_grains):
        t_sec = i * hop
        base_idx = int((i * read_stride) % max(1, len(src)))
        dry_grain, g_centroid, (gL, gR), jitter_offset, src_segment = _render_grain(src, base_idx, cfg, rng)
        if cfg.variant == "variant_a":
            wet_mono = np.zeros_like(dry_grain, dtype=np.float32)
        elif cfg.variant == "variant_b":
            # grains act as short IRs; source segment as exciter
            wet_mono = fftconvolve(src_segment, dry_grain, mode='full').astype(np.float32)
        else:
            idx = select_ir_index(
                strategy=cfg.ir_strategy,
                rng=rng,
                ir_items=ir_items,
                grain_centroid_hz=g_centroid if cfg.ir_strategy == 'centroid' else None,
                weights=weights,
                cycle_counter=cycle_counter if cfg.ir_strategy == 'cycle' else None,
            )
            ir = ir_items[idx].samples
            wet_mono = fftconvolve(dry_grain, ir, mode='full').astype(np.float32)
        wet_L = wet_mono * gL
        wet_R = wet_mono * gR
        start = int(sr * (t_sec + jitter_offset))
        start = max(0, start)
        end = min(len(out_l), start + len(wet_mono))
        wlen = end - start
        if wlen > 0:
            out_l[start:end] += wet_L[:wlen]
            out_r[start:end] += wet_R[:wlen]
        end_dry = min(len(dry_l), start + len(dry_grain))
        dlen = end_dry - start
        if dlen > 0:
            dry_l[start:end_dry] += dry_grain[:dlen] * gL
            dry_r[start:end_dry] += dry_grain[:dlen] * gR

    wet_mix_l = cfg.wet * out_l[:total_len]
    wet_mix_r = cfg.wet * out_r[:total_len]
    dry_mix_l = cfg.dry * dry_l[:total_len]
    dry_mix_r = cfg.dry * dry_r[:total_len]
    mix_l = wet_mix_l + dry_mix_l
    mix_r = wet_mix_r + dry_mix_r
    mix = np.stack([mix_l, mix_r], axis=-1)

    if cfg.normalize:
        peak = np.max(np.abs(mix)) + 1e-12
        if peak > 1.0:
            mix = mix / peak * 0.99
    return mix.astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, samplerate=sample_rate, subtype='PCM_24')


