from typing import Dict, Any, List


PRESETS: Dict[str, Dict[str, Any]] = {
    "Original-like": {
        "description": "Closer to the source; long grains, dense, subtle IR.",
        "grain_ms": 35,
        "density_hz": 100,
        "jitter": 0.02,
        "pitch_semitones": 1,
        "pan_spread": 0.2,
        "ir_ms": 10,
        "ir_strategy": "fixed",
        "wet": 0.3,
        "dry": 0.7,
    },
    "Airy shimmer": {
        "description": "Brighter, airy coloration with moderate motion.",
        "grain_ms": 15,
        "density_hz": 80,
        "jitter": 0.08,
        "pitch_semitones": 5,
        "pan_spread": 0.6,
        "ir_ms": 14,
        "ir_strategy": "weighted",
        "wet": 0.65,
        "dry": 0.35,
    },
    "Warm body": {
        "description": "Warmer mid-focused IR, smoother motion.",
        "grain_ms": 25,
        "density_hz": 60,
        "jitter": 0.05,
        "pitch_semitones": 2,
        "pan_spread": 0.3,
        "ir_ms": 20,
        "ir_strategy": "centroid",
        "wet": 0.5,
        "dry": 0.5,
    },
    "Sparse sparkles": {
        "description": "Sparse glitter with wider stereo and random color.",
        "grain_ms": 12,
        "density_hz": 30,
        "jitter": 0.12,
        "pitch_semitones": 7,
        "pan_spread": 0.9,
        "ir_ms": 12,
        "ir_strategy": "random",
        "wet": 0.7,
        "dry": 0.3,
    },
    "Percussive microroom": {
        "description": "Short grains and short IR for percussive micro-room feel.",
        "grain_ms": 10,
        "density_hz": 50,
        "jitter": 0.04,
        "pitch_semitones": 0,
        "pan_spread": 0.4,
        "ir_ms": 8,
        "ir_strategy": "fixed",
        "wet": 0.4,
        "dry": 0.6,
    },
}


def preset_names() -> List[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> Dict[str, Any]:
    return PRESETS[name]


