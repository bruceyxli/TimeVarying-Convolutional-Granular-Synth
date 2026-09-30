# Time-Varying Convolutional Granular Synth

A Python-based granular synthesizer that fuses **granular synthesis** with **time-varying convolution** using very short impulse responses (IRs). Timbre evolves at the grain rate without smearing attacks, enabling everything from evolving pads to percussive micro-rooms.

## ORBIT Windows VST3

The native C++/JUCE effect lives in [plugin/](plugin/README.md): live input,
per-grain convolution, parameter automation, session state, and a minimal ice-blue
interface with a live stereo input scope. The first native version implements
Standard mode; experimental A/B modes remain in the Python application.
See the plugin README for build, installation and dependency license details.
The **ORBIT Windows VST3** GitHub Actions workflow builds and validates Windows x64
artifacts. DAW-specific session testing remains a release requirement.

The lightweight offline web interface can be started with
`python -m src.app.web --port 56628`. Open the printed local address.

Both interfaces provide **Save** beside the preset selector: name the current sound,
then recall it from **User**. Presets include all parameters, including Reverb and
advanced settings; repeated names create numbered copies. The native plugin stores
files in `%APPDATA%\ORBIT\Presets` on Windows. The web version stores its own library
in this browser for the current site address; clearing site data removes it.
These are parameter presets, without source audio, and the two libraries are separate.
Hover over a parameter or its label for a short Chinese description. The web
interface also shows descriptions on keyboard focus; Escape dismisses the popup.
**Details** switches the main control area to a separate parameter page. **Back**
returns to the instrument with all current values preserved.

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

The most recent render remains available for playback and download after changing
controls. Press Render again to apply the new settings. Invalid uploads and render
parameters now produce readable errors rather than NaNs or silently generated noise.

## Performance and stability

The renderer reuses bounded resampling FIR caches, a per-render Hann window, and
per-render IR FFTs. IR-selection weights and centroids are prepared once; spectral
analysis is skipped unless centroid selection is active. Mixing accumulates directly
into one stereo output buffer. The UI caches decoded uploads and generated IR banks.

On one Windows 11 machine, five warm runs against `1f8f819` gave these medians
(48 kHz; seeded synthetic source; source/IR generation and WAV encoding excluded):

| Scenario | Before | After | Speedup | Traced peak memory before → after |
|---------|-------:|------:|--------:|---------------------------------:|
| Default, 10 s | 220.9 ms | 45.3 ms | 4.87× | 31.9 → 5.0 MiB |
| Dense, 40 s, 120 grains/s, 50 ms grains | 1925.4 ms | 455.2 ms | 4.23× | 125.3 → 19.1 MiB |
| Centroid selection, 10 s | 201.4 ms | 65.1 ms | 3.10× | 31.9 → 4.6 MiB |

The default cold render measured 212.0 → 64.1 ms. Memory was measured in separate
`tracemalloc` runs; it is **not total process RSS** and excludes previously cached
FIRs and input preparation. Full measurements, environment and other variants are in
[benchmark-windows.json](docs/benchmark-windows.json). Performance varies by machine.

These measurements describe the **offline Python renderer**, not the native VST3.
The Windows VST3 architecture and validation requirements are described in
[the VST3 migration plan](docs/vst3-roadmap.md).

### Audio behavior fixes

- Upward pitch shifts now read enough input to fill a grain. Previously +12 semitones
  left approximately 75% of a grain zero-padded. Pitch ratio quantization remains 0.01.
- Variant A now blends original and pre-convolved source **before granulation**;
  Wet=1 / Dry=0 produces audio, and Wet=0 / Dry=1 uses the original source. The source
  scan length stays equal to the original clip length; pre-convolution tails outside
  that clip are not scanned.
- Empty/non-finite sources and IRs, invalid weights and invalid parameters fail
  explicitly. Short valid inputs are zero-padded, never replaced with random noise.
- Output is exactly the requested duration, with later tails clipped. Normalization
  only attenuates peaks exceeding 1.0; it does not boost quieter renders.

These fixes intentionally change pitched renders and Variant A; old/new waveforms
are not claimed to be identical. The same inputs/configuration/seed remain reproducible.

### Tests and repeatable benchmarks

```bash
python -m unittest discover -s tests -v
python scripts/benchmark_render.py
python scripts/benchmark_render.py --compare-ref <baseline-commit> --output report.json
```

Tests cover all variants/selection strategies, pitch frequency and grain coverage,
FFT equivalence including tails, wet/dry linearity, repeatability, input immutability,
validation, audio decoding, and Streamlit result persistence. GitHub Actions runs
them on Windows/Linux with Python 3.10/3.12. Benchmarks use separate baseline/current
subprocesses and do not modify your checkout.

The API accepts sample rates 8–192 kHz, pitch ranges 0–24 semitones, grain lengths
up to 1000 ms, long IRs up to 10 s, and up to 1,000,000 grains / 115,200,000 output
samples per channel per render. These are allocation guards, not real-time targets.

## Project Structure

```
src/app/
├── engine.py     # Granular engine, variants, rendering pipeline
├── audio_io.py   # UI-independent byte decoding and sample-rate conversion
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

Original code: MIT — see [LICENSE](LICENSE). Native plugin builds also include
JUCE (AGPLv3 or commercial license) and Oxanium (SIL OFL); see [plugin licensing](plugin/README.md#licenses).
