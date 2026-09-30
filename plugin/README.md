# ORBIT VST3 — Windows x64

Native C++17/JUCE audio effect with the minimal, cold-blue ORBIT interface.
The left INPUT display receives live DAW input, with separate L/R envelopes,
fixed amplitude scale and clipping indication. No browser, Python process or
internet connection is needed by the compiled plugin.

## Install

Copy the **entire `ORBIT.vst3` directory** to a VST3 location scanned by your DAW,
typically `C:\Program Files\Common Files\VST3`, then rescan plugins. Insert ORBIT
as an audio effect on a track carrying audio. The companion `ORBIT.exe` is a
standalone audio-device host for testing, not the VST3 itself.

Start with Mix around 30–65% and moderate listening volume. Bypass is automatable;
both the native UI and the host bypass parameter transition to dry unity.

## Controls

- XY pad: horizontal Density, vertical Pitch Scatter. The adjacent sliders are
  keyboard-accessible alternatives; all controls notify the DAW for automation.
- Grain size: 5–50 ms; density: 10–120 grains/s; pitch: random ±0–12 semitones.
- Dry / Wet crossfades direct input with processed grains. There is no Render button.
- Details: trigger jitter, stereo spread, source lookback (0–200 ms), output gain,
  IR length (8/16/24/32 ms), five IR-selection strategies and a saved random seed.
- Five factory presets are starting points. Modified settings show Custom.
- Session state persists all parameters. Closing the editor does not stop processing.

## First native release scope

This implements the **Standard per-grain convolution** effect, with eight generated
micro-IRs per length. Variant A, Variant B, custom IR import and loaded-sample mode
remain in the Python reference only. This is a native interpretation, not a
sample-identical port: live grains read historical input, interpolation uses a
16-tap bandlimited table, IR generation/RNG differ, and Wet/Dry is a linked crossfade.

Zero host-reported latency means no extra block/FFT buffering delay. Grain lookback
and history capture are intentional audible time shifts; the wet signal is not
sample-aligned to the direct dry input. The output uses a smooth ceiling above
0.9 amplitude (except settled bypass), replacing offline peak normalization.

## Real-time design

Preparation allocates the history, 48 stereo voices, window/interpolation tables,
FFT plans, scratch and all IR spectra. `process()` allocates no memory and takes no
locks. Voices retain their convolved tails; overload drops a new grain rather than
cutting an existing one. Worst-case configured density fits the pool.

The scope uses a bounded SPSC queue. A slow/closed editor drops visual frames;
the audio thread never waits. The editor draws at 30 Hz and clears stale traces
when audio callbacks stop. Input continues to display when the host transport is
stopped but live input still arrives.

## Build and verify

Prerequisites: Visual Studio 2022 or later with Desktop development with C++ and
a Windows SDK, CMake 3.22+, and Git. Dependencies fetch pinned JUCE 8.0.6.

```powershell
cmake -S plugin -B plugin/build -A x64
cmake --build plugin/build --config Release --parallel 4
ctest --test-dir plugin/build -C Release --output-on-failure -V
```

Outputs: `plugin/build/Orbit_artefacts/Release/VST3/ORBIT.vst3` and
`plugin/build/Orbit_artefacts/Release/Standalone/ORBIT.exe`.
For DSP-only tests, configure with `-DORBIT_BUILD_PLUGIN=OFF`.

CI builds on Windows, runs deterministic/block-invariance and allocation tests,
prints dense 64-sample callback timing percentiles, then runs pluginval strictness 5.
Timing on a shared CI runner is diagnostic, not a hard-real-time guarantee. Actual
FL Studio/Ableton/REAPER session testing is still required before production use.

## Licenses

Original project code remains under the repository MIT license. Oxanium is bundled
under the SIL Open Font License in `web/fonts/OFL.txt`. JUCE 8 is available under
AGPLv3 or a separate commercial license; distributing a combined plugin requires
compliance with the applicable JUCE license, not just this repository's MIT license.
See https://juce.com/legal/juce-8-licence/ and the JUCE license included in the package.
