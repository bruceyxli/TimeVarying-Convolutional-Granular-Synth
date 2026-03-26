# Time-Varying Convolutional Granular Synth

A Python-based granular synthesizer that fuses **granular synthesis** with **time-varying convolution** using very short impulse responses (IRs). Timbre evolves at the grain rate without smearing attacks, enabling everything from evolving pads to percussive micro-rooms.

Built as a final project for **Music 159** (UC Berkeley).

## How It Works

```
Source Audio → Grain Renderer (window, pitch, pan, jitter)
            → Per-grain Convolution (micro IR)
            → Wet/Dry Mix → Normalization → Stereo WAV
```

Each grain is convolved with a different short IR (8–32 ms) selected by one of five strategies, making the convolution kernel **time-varying at the grain rate**.

## Variants

| Variant | Pipeline | Character |
|---------|----------|-----------|
| **Standard** | Per-grain short-IR convolution | Time-varying color, crisp onsets |
| **A** (Convolve → Granulate) | Full-source convolution with long IR, then granulate | Coherent ambience, less flicker |
| **B** (Grains as IR) | Grain acts as IR, source segment as exciter | Pronounced granular coloration |

## IR Selection Strategies

- **fixed** — always IR[0]
- **cycle** — round-robin
- **random** — uniform random
- **weighted** — random with configurable weight distribution
- **centroid** — picks IR whose spectral centroid is closest to the grain's

## Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repo
git clone https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth.git
cd TimeVarying-Convolutional-Granular-Synth

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run src/app/ui.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

### Usage

1. Optionally **upload audio** (WAV/FLAC/AIFF/OGG) — a built-in demo source is used if none is provided
2. Choose a **Variant** and adjust **Global** settings (sample rate, duration, wet/dry)
3. Shape **Grain** parameters (density, length, jitter, pitch range, pan spread)
4. Configure **IR** settings (length, bank size, selection strategy)
5. Click **Render**, preview, and **Download WAV**

Five built-in **presets** are available for quick starting points.

## Project Structure

```
src/app/
├── engine.py     # Granular engine, variants, rendering pipeline
├── features.py   # Spectral centroid computation
├── ir_bank.py    # Micro-IR generation, loading, and selection
├── presets.py    # Preset definitions
└── ui.py         # Streamlit GUI
```

## Key Techniques

- **Hann-windowed grains** to avoid clicks
- **Pitch shifting** via `resample_poly` with fixed output length
- **Per-grain spectral centroid** for content-aware IR selection
- **FFT convolution** (`fftconvolve`) — short IRs keep computation low
- **Equal-power panning** for stereo imaging
- **Reproducible** — all randomness is seeded (default seed: 2025)

## Tips for Musical Results

- **Denser space**: longer/more IRs + higher grain density
- **Clearer attacks**: shorter IR + lower Wet + moderate jitter
- **Wider stereo**: increase Pan Spread; keep Wet controlled to avoid wash
- **Adaptive timbre**: use "centroid" selection so IR color follows the grain's spectrum
- **Variant A** for coherent ambience; **Variant B** for pronounced granular coloration

## License

MIT License — see [LICENSE](LICENSE) for details.
