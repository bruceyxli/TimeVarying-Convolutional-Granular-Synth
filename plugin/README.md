# ORBIT VST3 — Windows x64

Native C++17/JUCE audio effect with the minimal, cold-blue ORBIT interface.
The left INPUT display receives live DAW input, with separate L/R envelopes,
fixed amplitude scale and clipping indication. The symmetric right OUTPUT display
shows a logarithmic spectrum of the actual final output, including Reverb and gain.
Click either display (or focus it and press Enter/Space) to cycle independently
through Waveform, Spectrum and Spectrogram. The small label shows the selected view.
Both sides use the same amplitude/frequency scales; spectrogram time moves right,
frequency rises upward, and brightness indicates level. View choices survive editor
reopening and session recall without changing sound presets.
No browser, Python process or
internet connection is needed by the compiled plugin.

Source: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth
The native implementation is on `codex/render-performance-stability` while PR #1
is under review. See `docs/vst3-validation.md` for the first verified build.

## Install

Copy the **entire `ORBIT.vst3` directory** to a VST3 location scanned by your DAW,
typically `C:\Program Files\Common Files\VST3`, then rescan plugins. Insert ORBIT
as an audio effect on a track carrying audio. The companion `ORBIT.exe` is a
standalone audio-device host for testing, not the VST3 itself.

Start with Mix around 30–65% and moderate listening volume. Bypass is automatable;
both the native UI and the host bypass parameter transition to dry unity.

## Controls

Hover over a parameter, its label or readout for about half a second to see a
short English description. The same help covers the XY pad, input scope, bypass
and advanced controls. Tooltips use the interface's dark, cool-colour palette.

- XY pad: horizontal Density, vertical Grain Size. The adjacent sliders are
  keyboard-accessible alternatives; all controls notify the DAW for automation.
- The three-position switch above the ring selects **PER GRAIN**, **PRE CONV** (A)
  or **GRAIN IR** (B). The existing Details selector remains synchronized, including
  host automation and preset recall. The ring morphs from ripples to elliptical
  flow to a three-lobed shape; its white/blue colour still depends only on Reverb.
- The four main parameters use matching large numeric readouts, units and faders.
  Double-click a number (or focus it and press Enter) to type. Enter confirms;
  Escape/clicking away cancels. Values clamp to the parameter range and snap to its
  step; Dry/Wet is entered as 0–100%. Host automation and presets update the readouts.
- Grain size: 5–50 ms; density: 10–120 grains/s; pitch: random ±0–12 semitones.
- Dry / Wet crossfades direct input with processed grains. There is no Render button.
- Reverb adds a damped stereo room after the granular mix. Its dedicated compact
  dial colours the halo: zero is white without glow, increasing amounts blend
  toward ice blue with a soft halo. Bypass shows white. Density controls line count,
  Grain Size the ring width, Pitch its deformation and Dry/Wet its definition.
- Main faders have etched divisions and larger caps; Shift-drag adjusts precisely,
  double-click restores the parameter default, and wheel scrolling leaves values alone.
- Details: trigger jitter, stereo spread, source lookback (0–200 ms), output gain,
  IR length (8/16/24/32 ms), five IR-selection strategies and a saved random seed.
- Five factory presets are starting points. Modified settings show Custom.
- Session state persists all parameters. Closing the editor does not stop processing.
- **Save** beside the preset selector opens a small naming panel. Enter a name and
  press Save/Enter; the preset appears under **User**. Escape/Cancel closes the panel.
  All 15 parameters are saved, including signal path, Long IR, Reverb, output gain, seed and bypass.
  Same-name saves create a numbered copy. User presets persist across DAW restarts
  as `.orbitpreset` files in `%APPDATA%\ORBIT\Presets` on Windows (up to 256).
  The library refreshes when the selector opens, including saves by other instances.
  Factory presets now recall complete settings; modified values display Custom.
- Existing 0.2 sessions load with Reverb off. The new parameter is appended without
  changing existing IDs/order. The host tail allowance is now 3 seconds.

## Native signal paths (0.5.0)

In **Details → Convolution → Signal path**, choose:

- **Per-grain convolution**: each grain uses one of eight generated micro-IRs per
  length, with the existing five selection strategies.
- **Convolve → Granulate** (A): a generated long IR processes the live input before
  grain capture. Long IR ranges from 40–300 ms; changes interpolate prepared kernels.
- **Grains as IR** (B): each pitched, windowed grain becomes an impulse response
  excited by a raw historical source segment. Per-channel L1 normalization controls gain.

IR length/selection appear only for Standard; Long IR appears only for A. All modes
use the same final Dry/Wet crossfade, followed by Reverb and output gain. In particular,
native A retains the plugin's direct dry endpoint; Python A blends before granulation.
Old sessions and version-1 user presets migrate to Standard / 120 ms without
changing existing parameter IDs or order. New presets use format version 2.

Custom IR import and loaded-sample mode remain in the Python reference only.
This is a native interpretation, not a
sample-identical port: live grains read historical input, interpolation uses a
16-tap bandlimited table, IR generation/RNG differ, and Wet/Dry is a linked crossfade.

Host-reported latency remains zero: the direct dry path has no added delay. A adds
256 samples of convolution buffering to its creative wet path. Grain lookback
and history capture are intentional audible time shifts; the wet signal is not
sample-aligned to the direct dry input. The output uses a smooth ceiling above
0.9 amplitude (except settled bypass), replacing offline peak normalization.

## Real-time design

Preparation allocates the history, 48 stereo voices, window/interpolation tables,
FFT plans, scratch and all IR spectra (including 53 long-IR lengths). `process()` allocates no memory and takes no
locks. Voices retain their convolved tails; overload drops a new grain rather than
cutting an existing one. Worst-case configured density fits the pool.

The scope and output spectrum use bounded SPSC queues. A slow/closed editor drops visual frames;
the audio thread never waits. The editor draws at 30 Hz and clears stale traces
when audio callbacks stop. Input continues to display when the host transport is
stopped but live input still arrives.
The 4096-point spectrum FFT runs on the editor thread and combines stereo energy
without cancelling opposite-phase channels. Its 64 bands cover 20 Hz to the lesser
of 20 kHz and Nyquist, with a fixed -90 dBFS floor and smooth decay.

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
