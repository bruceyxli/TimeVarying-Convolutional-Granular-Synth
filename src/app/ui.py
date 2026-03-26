import io
from typing import Optional, List

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import soundfile as sf
import streamlit as st

from src.app.engine import GranularConfig, render_offline
from src.app.ir_bank import generate_demo_ir_bank, IRItem
from src.app.presets import preset_names, get_preset


st.set_page_config(
    page_title="Time-Varying Convolutional Granular Synth",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
st.title("Time-Varying Convolutional Granular Synth")


# UI Frame for Streamlit
st.markdown(
    """
    <style>
      /* Keep container width and padding minimal customization */
      [data-testid="stAppViewContainer"] .main .block-container { max-width: 1280px; padding-top: 1rem; padding-bottom: 1rem; }
      /* Hide Streamlit top toolbar, deploy button, and overflow menu */
      [data-testid="stToolbar"] { display: none !important; }
      header { visibility: hidden; height: 0; }
      footer { visibility: hidden; }
      #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Default parameters
DEFAULTS = {
    "sr": 48000,
    "duration": 10.0,
    "wet": 0.6,
    "dry": 0.4,
    "variant": "standard",
    "long_ir_ms": 120.0,
    "density": 60,
    "grain_ms": 20,
    "jitter": 0.1,
    "pitch": 7,
    "pan": 1.0,
    "ir_len": 16,
    "num_irs": 64,
    "ir_strategy": "weighted",
    "seed": 2025,
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Apply any pending duration update BEFORE widgets are created
_pending_dur = st.session_state.get("_desired_duration", None)
if _pending_dur is not None:
    try:
        st.session_state["duration"] = float(_pending_dur)
    except Exception:
        pass
    st.session_state.pop("_desired_duration", None)

def _read_uploaded_audio(file, target_sr: int) -> np.ndarray:
    if file is None:
        return np.array([], dtype=np.float32)
    data, sr = sf.read(file, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(target_sr, sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)
    return data


with st.sidebar:
    st.header("Presets")
    sel_preset = st.selectbox("Preset", ["None"] + preset_names(), index=0)
    if sel_preset != "None":
        desc = get_preset(sel_preset)["description"]
        st.caption(desc)
        if st.button("Apply preset"):
            # Preserve existing duration explicitly
            _prev_duration = float(st.session_state.get("duration", 10.0))
            p = get_preset(sel_preset)
            st.session_state["density"] = p["density_hz"]
            st.session_state["grain_ms"] = p["grain_ms"]
            st.session_state["jitter"] = p["jitter"]
            st.session_state["pitch"] = p["pitch_semitones"]
            st.session_state["pan"] = p["pan_spread"]
            st.session_state["ir_len"] = p["ir_ms"]
            st.session_state["ir_strategy"] = p["ir_strategy"]
            st.session_state["wet"] = p["wet"]
            st.session_state["dry"] = p["dry"]
            # Restore duration (preset must not change it)
            st.session_state["duration"] = _prev_duration
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    st.header("Global")
    variant_map = {
        "Standard": "standard",
        "Convolve → Granulate (A)": "variant_a",
        "Grains as IR (B)": "variant_b",
    }
    variant_names = list(variant_map.keys())
    _var_idx = variant_names.index({"standard":"Standard","variant_a":"Convolve → Granulate (A)","variant_b":"Grains as IR (B)"}[st.session_state.get("variant","standard")])
    variant_label = st.selectbox("Variant", variant_names, index=_var_idx, key=None)
    st.session_state["variant"] = variant_map[variant_label]
    if st.session_state["variant"] == "variant_a":
        st.slider("Long IR Length (ms)", 40, 300, int(st.session_state.get("long_ir_ms",120)), 5, key="long_ir_ms")
    _sr_options = [48000, 44100]
    _sr_index = _sr_options.index(st.session_state.get("sr", 48000))
    sr = st.selectbox("Sample Rate", _sr_options, index=_sr_index, key="sr")
    duration = st.slider("Duration (sec)", 2.0, 40.0, st.session_state.get("duration", 10.0), 1.0, key="duration")
    wet = st.slider("Wet", 0.0, 1.0, st.session_state.get("wet", 0.6), 0.05, key="wet")
    dry = st.slider("Dry", 0.0, 1.0, st.session_state.get("dry", 0.4), 0.05, key="dry")
    st.divider()
    st.header("Grain")
    density = st.slider("Density (grains/s)", 10, 120, st.session_state.get("density", 60), 1, key="density")
    grain_ms = st.slider("Grain Length (ms)", 5, 50, st.session_state.get("grain_ms", 20), 1, key="grain_ms")
    jitter = st.slider("Trigger Jitter", 0.0, 0.5, st.session_state.get("jitter", 0.1), 0.01, key="jitter")
    pitch_semi = st.slider("Pitch Range (±semitones)", 0, 12, st.session_state.get("pitch", 7), 1, key="pitch")
    pan_spread = st.slider("Pan Spread", 0.0, 1.0, st.session_state.get("pan", 1.0), 0.05, key="pan")
    st.divider()
    st.header("IR")
    ir_len = st.slider("IR Length (ms)", 6, 32, st.session_state.get("ir_len", 16), 1, key="ir_len")
    num_irs = st.slider("IR Bank Size", 8, 128, st.session_state.get("num_irs", 64), 1, key="num_irs")
    _ir_opts = ["weighted", "centroid", "random", "cycle", "fixed"]
    _ir_idx = _ir_opts.index(st.session_state.get("ir_strategy", "weighted"))
    ir_strategy = st.selectbox("IR Selection", _ir_opts, index=_ir_idx, key="ir_strategy")
    seed = st.number_input("Random Seed", value=st.session_state.get("seed", 2025), step=1, key="seed")

st.subheader("Input")
uploaded = st.file_uploader("Upload audio (optional; built-in demo used if none)", type=["wav", "flac", "aiff", "aif", "ogg"])

# Auto-align Duration to uploaded audio length (on first detect or SR change)
if uploaded is not None:
    try:
        # Ensure file pointer at start for multiple reads
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
    except Exception:
        pass
    _last_name = st.session_state.get("_last_uploaded_name")
    _last_sr = st.session_state.get("_last_uploaded_sr")
    _name = getattr(uploaded, "name", None)
    # Recompute duration if new file or sample rate changed
    if _name != _last_name or _last_sr != st.session_state.get("sr", 48000):
        data = _read_uploaded_audio(uploaded, target_sr=st.session_state.get("sr", 48000))
        if len(data) > 0:
            dur = float(len(data) / float(st.session_state.get("sr", 48000)))
            # Clamp to slider's bounds (2..40 seconds)
            if np.isfinite(dur) and dur > 0:
                new_dur = float(np.clip(dur, 2.0, 40.0))
                if abs(new_dur - float(st.session_state.get("duration", 10.0))) >= 1e-6:
                    # Defer update until next run to avoid modifying after widget creation
                    st.session_state["_desired_duration"] = new_dur
                st.session_state["_last_uploaded_name"] = _name
                st.session_state["_last_uploaded_sr"] = st.session_state.get("sr", 48000)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

do_render = st.button("Render", type="primary")

if do_render:
    cfg = GranularConfig(
        sample_rate=sr,
        duration_sec=float(duration),
        variant=str(st.session_state.get("variant","standard")),
        long_ir_ms=float(st.session_state.get("long_ir_ms",120.0)),
        density_hz=float(density),
        grain_ms=float(grain_ms),
        jitter=float(jitter),
        pitch_semitones=float(pitch_semi),
        pan_spread=float(pan_spread),
        wet=float(wet),
        dry=float(dry),
        ir_strategy=ir_strategy,
        rng_seed=int(seed),
        normalize=True,
    )
    # Source material
    source = _read_uploaded_audio(uploaded, target_sr=sr)
    if len(source) == 0:
        # Generate default demo source
        n = int(sr * duration)
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0.0, 0.05, n).astype(np.float32)
        time_vec = np.arange(n) / sr
        freqs = [220.0, 277.18, 329.63]
        pad = sum(np.sin(2 * np.pi * f * time_vec) for f in freqs) / 3.0
        env = np.clip(np.linspace(0, 1, int(0.5 * sr)), 0, 1)
        env = np.pad(env, (0, max(0, n - len(env))), mode='edge')
        pad = (pad * env).astype(np.float32)
        source = (pad * 0.3 + noise).astype(np.float32)

    # IR bank (auto-generated)
    with st.spinner("Generating micro IR bank..."):
        ir_items: List[IRItem] = generate_demo_ir_bank(
            num_irs=int(num_irs),
            target_ir_ms=float(ir_len),
            sample_rate=sr,
            seed=int(seed),
        )

    with st.spinner("Rendering..."):
        if ir_strategy == "weighted":
            # simple linear weights demo
            weights = np.linspace(1.0, 2.0, num=len(ir_items)).astype(np.float64)
        else:
            weights = None
        audio = render_offline(source_audio=source, ir_items=ir_items, cfg=cfg, weights=weights)

    # Play & Download (encode to WAV bytes first to avoid internal conversion issues)
    st.success("Done")
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=sr, format="WAV", subtype="PCM_24")
    wav_bytes = buf.getvalue()
    st.audio(wav_bytes, format="audio/wav")
    # Build Windows-safe timestamped filename using uploaded sample name (or demo)
    from datetime import datetime
    ts = datetime.now().strftime("%m-%d %H-%M-%S")
    sample_name = "demo"
    try:
        if uploaded is not None and hasattr(uploaded, "name") and uploaded.name:
            import os
            sample_name = os.path.splitext(uploaded.name)[0]
    except Exception:
        pass
    dl_name = f"[{ts}] {sample_name}.wav"
    st.download_button("Download WAV", data=wav_bytes, file_name=dl_name, mime="audio/wav")

st.markdown("---")


