"""UI-independent decoding; byte input avoids shared file-cursor state."""
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def decode_audio(data: bytes, target_sr: int) -> np.ndarray:
    if not isinstance(target_sr, (int, np.integer)) or not 8000 <= target_sr <= 192000:
        raise ValueError("target_sr must be an integer between 8000 and 192000")
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Uploaded audio is empty or contains non-finite samples")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != target_sr:
        divisor = gcd(int(target_sr), sr)
        audio = resample_poly(audio, target_sr // divisor, sr // divisor)
    if not np.all(np.isfinite(audio)):
        raise ValueError("Uploaded audio overflowed during conversion")
    return audio.astype(np.float32, copy=False)
