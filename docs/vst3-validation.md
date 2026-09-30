# ORBIT native validation — 2026-09-30

## 0.3.0 — precision faders and Reverb halo

- Native source commit: `17cdfa4399e2baf1b04857e65c3d08c30ccd9cc0`.
- Successful Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36786088397
- VST3 module: 7,178,752 bytes; SHA-256:
  `E05E05D5CAE5C2346C64E20708CAE46F6BF6DC5210584FAB74C127B2BA699A0B`.
- Four redesigned faders, fine dragging, default reset, smoothly parameter-driven
  ring geometry and a dedicated Reverb dial. Native white/off and blue/80% images
  were rendered and visually checked; matching web states were checked in-browser.
- Reverb adds a prepared, damped stereo room after the granular mix and before output
  gain/ceiling. Amount zero preserves the old path; bypass smoothly reaches dry unity.
- Native regression tests now cover a real stereo tail, decay, exact Reverb block-size
  invariance, zero-amount transparency, bypass, sample-rate extremes, automated amount
  changes and state recall. No callback C++ heap allocations detected.
- pluginval 1.0.4 strictness 5 passed in CI and locally for the actual 0.3.0 VST3.
- Local dense benchmark with Reverb at 100%, same setup described below:
  p50 **2.1 us**, p95 **231.7 us**, p99 **271.7 us**, max **450.0 us**;
  **0 / 3000** blocks exceeded 1333.33 us. This is a short measurement, not a guarantee.
- 23 Python tests passed, including independent sample-by-sample checking of the
  offline room impulse response, zero amount, stereo decay and input validation.
  A 10-second web render with Reverb at 80% completed and exposed a downloadable WAV.

Reverb is independent of the existing grain presets. It defaults off and is appended
to the native parameter list; old state without it is migrated to zero. The web uses
a cached three-second room impulse response and offline convolution, while the native
plugin runs the corresponding delay network continuously. The existing Python
renderer and native granular engine still differ as documented in the plugin README.

Real DAW session/soak testing remains required. No new claims of production
certification or sample-identical Python/native granular output are made.

## 0.2.0 baseline

This is an initial Windows x64 VST3 build, not a completed DAW certification.

- Binary source commit: `1822d0fdfa9faf74fe6780c74b81c6c1d484aac8`.
- Successful build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36777837455
- JUCE 8.0.6, Release x64, static MSVC runtime.
- VST3 module: 7,169,536 bytes; standalone: 7,972,352 bytes.
- VST3 module SHA-256: `3328D110F1308BB387679E2DC678E05892235B36404E8D723CA3E70FA1BC94AB`.

## Completed checks

- Native FFT roundtrip and convolution against direct convolution, including tails.
- Exact block-size invariance for all five IR selectors and seeded reset.
- Dry endpoint, settled bypass, silence, opposite-phase stereo and scope queue overflow.
- Finite bounded processing at 44.1/48/96/192 kHz, dense settings and automation.
- No C++ heap allocations detected in tested DSP callback paths.
- Host state save/restore, malformed-state rejection, oversized mono input blocks,
  native editor creation, rendering and reopening.
- Tracktion pluginval 1.0.4 strictness 5 passed in CI and on the local Windows machine,
  including 44.1/48/96 kHz, 64–1024 sample blocks, automation, state and editor tests.
  Steinberg's separate validator was not run.
- Existing 19 Python regression tests passed locally and in Windows/Linux CI.

## Local callback timing

AMD Ryzen 7 9800X3D, Windows, one DSP instance without a DAW or editor. Recorded
3,000 contiguous 64-sample blocks at 48 kHz (4 seconds of synthetic stereo input).
Settings: 120 grains/s, 50 ms grains, 32 ms IR, ±12 st scatter, centroid selection.

| Metric | Microseconds |
|---|---:|
| p50 | 1.2 |
| p95 | 229.3 |
| p99 | 293.3 |
| Maximum | 500.8 |
| Block deadline | 1333.33 |

Observed deadline misses: **0 / 3000**. This short, single-instance measurement is
not a hard real-time guarantee. Grain-trigger blocks do substantially more work
than blocks that only mix existing voices. High sample rates, smaller buffers,
heavy automation, multiple instances and long sessions require further measurement.

## Still required before a production release

Real FL Studio/Ableton/REAPER sessions: playback, record monitoring, project reload,
automation, bounce, bypass, sample-rate changes, GUI resizing and long-session load.
Also pending: measured listening/reference comparisons, Variant A/B and custom IR
import. The native Standard effect intentionally differs from Python; see the
plugin README for the source-history, interpolation, RNG and output-ceiling changes.
