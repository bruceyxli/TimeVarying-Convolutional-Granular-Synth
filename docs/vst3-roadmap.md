# Windows VST3 migration plan

The target is a Windows x64 VST3 built with C++/JUCE. The Python application
is the offline reference engine. The first native implementation now lives in
`plugin/`: Standard mode, preallocated C++ DSP, JUCE host integration, automated
Windows builds and the cold-blue ORBIT editor with live stereo input display.
See `plugin/README.md` for implemented scope and intentional sound differences.
Keep the reference renderer and regression tests during the port instead of embedding
Python/Streamlit in the audio callback.

## 1. Establish the sound contract

- Preserve Hann grains, equal-power pan, per-grain IR changes and three variants.
- Standard is the first porting target. Verify A/B after Standard is stable.
- Keep seeded offline fixtures for pitch, IR selection, stereo mixing and tails.
  C++ and NumPy RNG algorithms must either match explicitly or share a stored grain
  event schedule; matching a seed alone is not a cross-language equivalence test.
- Offline normalization sees the whole render. Replace it in live processing with
  output gain and optional limiting, documenting latency and the sonic difference.
- Preserve convolution tails belonging to each active grain when another IR is selected.

## 2. Separate preparation from processing

Build a UI-independent C++ engine with `prepare(sampleRate, maxBlockSize)`, `reset()`
and `processBlock(input, output, sampleCount)` interfaces. The host sets sample rate
and block size. A JUCE AudioProcessor owns the engine; the editor owns no DSP state.

Preparation/worker-thread responsibilities:

- Decode/resample source clips and IRs; validate input and parameter ranges.
- Build resampling polyphase tables, windows, FFT plans and IR spectra.
- Allocate a bounded grain voice pool, delay/ring buffers and convolution scratch.
- Transfer prepared immutable resources to the audio thread through a bounded handoff.
  Reclaim old resources on the worker thread after the callback stops using them;
  a shared pointer destructor must not free large buffers inside the callback.

Audio-thread responsibilities:

- Read parameter snapshots without locks, schedule events with absolute sample positions,
  and update active grains. Processing must not depend on where the host splits blocks.
- Perform no heap allocation, file I/O, filter design, logging, UI calls or waiting.
- Use bounded overload handling. The first implementation drops new events when
  its 48 voices are occupied, preserving existing tails without a discontinuity.
- Smooth continuous controls; change FFT/grain resources at defined boundaries.
- Handle host bypass, suspend/resume, transport reset, sample-rate changes and denormals.

## 3. Define the VST3 input mode

Start as an **audio effect**: live input feeds a preallocated history ring. The offline
renderer scans a complete file, including future samples; live input cannot do that.
Specify the live source lookback/read-position control and under-run behavior explicitly.
At +24 semitones a grain consumes roughly four times its output length, which sets
the history-buffer requirement. An optional loaded-sample mode can retain offline
scan semantics. A MIDI instrument would be a separate product mode, not an implicit port.

Variant A requires streaming partitioned convolution before granulation. Standard
uses per-grain convolution; Variant B uses each prepared grain as its kernel.
Measure direct versus FFT convolution for short kernels before selecting the C++ path.
Report actual processing latency to the host and align dry/wet paths accordingly.

### Left-side live input display

The left waveform window in the VST3 effect must show the actual DAW input in real
time, before granular processing and wet/dry mixing. This replaces the static demo
or uploaded-file preview used by the current offline webpage.

- Keep the minimal ORBIT layout: one small `INPUT` label and a scrolling waveform;
  no explanatory copy or required file import in live effect mode.
- Show approximately the most recent second of input with a fixed amplitude scale.
  Preserve stereo activity using separate L/R traces, rather than summing channels
  and hiding opposite-phase signals. Indicate clipping subtly at the display edge.
- Display real incoming samples, not a decorative animation. Silence scrolls to a
  flat baseline; if callbacks stop, clear stale audio after the display window elapses.
  Host transport stopping must not suppress monitoring if live input still arrives.
- Reduce samples to min/max envelopes on the audio thread, pushing them into a
  preallocated single-producer/single-consumer queue without allocating or blocking.
  If the UI falls behind, drop visualization data rather than delay audio processing.
- The editor consumes the envelopes and paints at approximately 30 fps. Display
  buffering adds no audio-path delay. Opening/closing the editor and changing sample
  rates must safely reset or reconnect the display without affecting DSP.
- Validate mono/stereo input, opposite-phase stereo, silence, clipping, stopped
  callbacks, live input with stopped transport, and editor reopen behavior.

This is a requirement for the C++ VST3 implementation; the current webpage remains
an offline renderer and must not label its static source preview as live input.

## 4. Parameters and saved state

| Existing field | VST3 mapping |
|---|---|
| Density, grain length, jitter, pitch range, pan, Wet, Dry | Stable parameter IDs; host automation; smoothing |
| Variant, IR strategy | Discrete automatable choices with transition rules |
| IR length/bank size, source files | Worker preparation; asynchronous completion |
| Sample rate | Host-owned, not a user automation parameter |
| Duration | Offline export only |
| Normalize | Offline export only; live output gain/limiter instead |
| Seed | Persisted deterministic state with an explicit reset policy |

Serialize versioned state using JUCE's parameter/state facilities. Save resource
identity and restore missing-file errors without blocking the audio thread.

## 5. Release gates

1. DSP regression comparison against the Python reference, with intentional changes
   documented and bounded numerical tolerances.
2. Block-size invariance at 32/64/128/256/512/1024 samples and changing block sizes.
3. Callback p50/p95/p99/max duration and deadline-miss counts at 44.1/48/96 kHz,
   with dense settings, automation, bank swaps, many instances and long sessions.
   At 48 kHz a 64-sample block provides about 1.33 ms; offline throughput does not
   prove that every callback meets this deadline.
4. Allocation/lock auditing in the callback, CPU/memory soak testing and bounded
   overload behavior.
5. VST3 validation plus FL Studio, Ableton Live and REAPER checks: scanning, state
   restore, bypass, automation, offline export and project reload.

The first native implementation covers Standard DSP, block-processing regression
tests, allocation checks, state/mono/editor tests and pluginval validation in CI.
Remaining release work includes real DAW session/soak checks, broader callback
timing measurements and a measured sound comparison with the Python reference.
Variant A/B and custom IR import remain later milestones.
