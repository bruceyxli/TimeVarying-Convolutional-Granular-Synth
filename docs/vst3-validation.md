# ORBIT native validation — 2026-09-30

## 0.6.0 — switchable views, main mode switch and circular XY

- Native source commit: `4f627c79cb6b367cfda44fa0b0e59dd6b0fd1066`.
- Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36802415108
- VST3/standalone build, DSP/host checks and pluginval 1.0.4 strictness 5 passed.
  Host checks also passed on the local Windows machine; native main-page screenshots
  for all three convolution modes were inspected.
- VST3 SHA-256: `818E7A9E24F25538967222D5AD1EEDFD30FC28DB7C561A667EA95D3AA7111119`.
- Both clickable displays cycle independently through Waveform, Spectrum and
  Spectrogram. The native input is captured before processing and output after
  gain/mono summing. Both use identical frequency/amplitude scales. The spectrogram
  uses a bounded 192 x 64 circular image with no image shifting; FFT and image work
  stay on the editor thread. Idle drawing stops once the display has settled.
- Presentation state survives editor/session recall and remains outside the 15
  sound parameters and sound presets. Tests cover state recall, independent cycling,
  wraparound and unchanged sound parameters. Native spectrogram snapshots use a
  known tone fixture; web screenshots use actual demo playback.
- PER GRAIN / PRE CONV / GRAIN IR directly set the existing automatable variant
  parameter. Details and preset recall stay synchronized. Native tests exercise all
  three main buttons, parameter values, conditional Details controls and rendering.
  Shape transitions blend ripples, elliptical flow and three lobes. Reverb alone
  controls white-to-blue colour/glow.
- The XY handle now uses a square-to-disk mapping and inverse, with radial clipping
  outside the field. C++ and JavaScript grid tests cover all parameter quadrants,
  full-range round trips, centre/extremes and outside projection. A browser drag
  beyond the boundary was inspected: the handle stayed on the circular field.
- Web input preview uses the same decoded source as the renderer, played through a
  silent analysis branch. It loops alongside rendered output, not at each granular
  read position. Waveform shows a labelled full-clip overview when paused and a
  short live stereo window during playback. Native waveform history is one second;
  web live history is 4096 samples. These are intentionally different preview modes.
- 26 Python tests and seven Node tests passed, including actual demo/imported source
  preview bytes, missing-source errors, geometry and preset persistence. Browser
  checks cover both plot controls, Enter/Space, view persistence after reload,
  source/output spectrogram playback, saved preset recall and main/Details mode sync.
  No browser console errors were observed.

Audio algorithms and sound parameter IDs/order are unchanged from 0.5.0. The extra
input capture is bounded and nonblocking. Real DAW session/soak validation remains
pending; previous short DSP benchmarks are not a host-performance guarantee.

## 0.5.0 — three native IR paths and output spectrum

- Native source commit: `4889cd7056e09b85af45fad4af2ab18d25a010b0`.
- Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36797031444
- Release x64 VST3/standalone build, both native test suites and pluginval 1.0.4
  strictness 5 passed in CI. DSP and host checks also passed locally.
- VST3 SHA-256: `EBEC2A6718D9F551BA4CD94ABE0F0A53317103918D4ECF08D5C93B2C363EA42A`.
- Variant A's partitioned convolution matches independent direct convolution at
  40, 125 and 300 ms, including its 256-sample wet-path delay and complete tail.
  Changing Long IR changes audio; leaving/returning to A does not replay stale history.
- A/B pass exact block invariance, silence, non-silent wet output, direct dry and
  bypass checks. B's source-derived convolution passes the double-polarity inversion
  identity. All three paths remain finite/bounded at 44.1/48/96/192 kHz; mode and
  Long IR automation were exercised. The tested audio paths allocate no heap memory.
- Session recall includes both appended parameters. Old sessions and version-1
  presets migrate to Standard / 120 ms; incomplete version-2 presets are rejected.
  Mode-specific native controls and all three Details screenshots were checked.
- Spectrum tests recover a known antiphase stereo tone's frequency and level,
  reject distant bands, reset on sample-rate changes and silence, and exercise full
  queue recovery. Host checks compare captured spectrum frames to the actual final
  mono output after gain. FFT analysis runs on the editor thread.
- Web: actual rendered playback drives the symmetric right spectrum; left waveform
  remains intact. Playback, return to silence, all three mode controls and existing
  user-preset recall were checked. No browser console errors. Five preset tests and
  23 Python tests passed. Web FFT uses the browser's analyzer window, so spectral
  heights are not calibrated identically to the native Hann-window display.

Local AMD Ryzen 7 9800X3D, one DSP instance, 48 kHz, 64 samples, 3,000 blocks per
mode; 120 grains/s, 50 ms grains, ±12 st, 32 ms micro-IR, 300 ms Long IR, full Reverb:

| Mode | p50 (us) | p95 (us) | p99 (us) | Max (us) | Deadline misses |
|---|---:|---:|---:|---:|---:|
| Standard | 2.2 | 229.5 | 296.4 | 446.5 | 0 / 3000 |
| A | 2.6 | 120.7 | 171.9 | 215.6 | 0 / 3000 |
| B | 2.2 | 625.1 | 708.5 | 906.4 | 0 / 3000 |

The block deadline is 1333.33 us. B has the highest burst cost. These short DSP-only
measurements exclude the host/editor and do not establish high-rate, multi-instance
or long-session safety. Actual DAW session/soak and listening comparisons remain
pending. Native modes retain a common direct Dry/Wet endpoint and differ from the
offline reference in normalization, historical source capture and generated IRs.
Custom IR import remains outside the native build.

## 0.4.3 — English descriptions

- Native source commit: `c1678ba4cd84bab468a42fed1add23a5e98a5678`.
- Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36792998379
- Build, DSP/host checks and pluginval 1.0.4 strictness 5 passed in CI.
- VST3 SHA-256: `A89D77B4C9B182FE10FCB49C5F73E3A9C318FF720EA3E5E3FC0095347D691017`.
- All 15 native and 16 web parameter descriptions now use English, including
  advanced settings and XY help. Native input-scope and bypass help are English too.
- Checked application source for remaining Chinese descriptions; JavaScript syntax
  and browser tooltip text/wrapping were verified. User-provided preset names remain
  unchanged. This update changes copy only; DSP and parameter IDs are unchanged.

## 0.4.2 — shared header navigation

- Native source commit: `0d1e6665bca4ddbe59bb8809e53b836a1e7f3a18`.
- Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36790596728
- Build, DSP/host checks and pluginval 1.0.4 strictness 5 passed in CI.
  The packaged native Details screenshot was inspected.
- VST3 SHA-256: `0C8951F866C2D191C171F32642EB2B9A74D71EB54AA9D49BE449DFF9FB1E5DF5`.
- Details / Back now occupies the same upper-right header position in both views.
  Native Bypass sits to its left; the old lower/left navigation buttons are removed.
- Browser checks confirmed both navigation states and parameter preservation.
  Invalid advanced values correctly reopen Details for correction before rendering.
- This is a layout change; DSP and parameter IDs are unchanged.

## 0.4.1 — parameter help and Details page

- Native source commit: `3684876abfd7abe26ace980eb739e7fc72a7f2da`.
- Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36789562554
- VST3 SHA-256: `759C816E8DA5E2BB20D6EFFEAAAC0ED9F020A743F972F43096C3D5F7E891B39A`.
- Build, native DSP/host checks and pluginval 1.0.4 strictness 5 passed in CI.
  Host checks also passed locally; the rendered native Details page was inspected.
- Short Chinese tooltip descriptions cover parameters, labels, readouts and XY.
  Native tooltips use a 550 ms delay and the existing cold-colour theme.
- Details now replaces the main control area. Back restores the main page;
  the native host window retains its size. Header preset controls remain available.
- Browser checks confirmed the main controls disappear from the Details page,
  edited Jitter survives a round trip, and Density, Grain, Pitch, Wet and Reverb
  retain their values. Help text, keyboard focus and Escape dismissal were checked.
- Native host checks cover page navigation, unchanged window bounds and complete
  parameter preservation, and also render the Details page for visual inspection.
- DSP and parameter IDs are unchanged. Real DAW session/soak testing remains pending.

## 0.4.0 — user presets

- Native source commit: `d3c8458a6a197a2106fb2eae2dc09e7484134126`.
- Successful Windows build: https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36787966208
- VST3 SHA-256: `1BCD7A62033845085C16BBDCA5039B09E85F5CE2A933E9C731054FC874AEC06B`.
- Save/name/recall controls added without changing DSP or parameter IDs. Native
  presets contain all 13 parameters; factory recall now also resets Reverb,
  lookback, output, seed and bypass to the factory defaults.
- Native tests passed in CI and locally: Unicode names, numbered duplicate copies,
  reload through a new store, all-parameter recall, malformed-file rejection and
  atomic rejection of invalid parameters. File access remains on the UI thread.
- Existing DSP tests passed in CI; pluginval 1.0.4 strictness 5 passed in CI and
  locally on the actual VST3. Native editor rendering/reopen passed locally.
- Five Node preset-store tests and 23 Python tests passed locally. In-browser
  testing saved a Chinese-named preset, changed parameters, reloaded the page and
  restored Density 71, Grain 25 ms, Pitch 6.7 st, Wet 50% and Reverb 100%.
  Modified preset labels and the naming dialog were also checked.
- The web library stores 15 offline controls in localStorage; it is separate from
  the native per-user file library. Presets contain settings, not source audio.
- Real DAW session/soak testing is still pending.

## 0.3.1 — luminous XY cursor

UI-only update: graduated optical bloom, a highlighted core, segmented locator ring,
and hover/drag feedback. The point still follows Reverb colour. DSP and parameter
semantics are unchanged. Web JavaScript syntax and browser rendering were checked.

- Native source: `0c3720f26f285acbaa2fb2804b43f11308cb4346`.
- Successful build, native regression/host tests and pluginval strictness 5:
  https://github.com/bruceyxli/TimeVarying-Convolutional-Granular-Synth/actions/runs/36787041042
- Native editor paint/reopen checks also passed locally; rendered white/blue views
  were inspected. No additional real DAW session testing was performed.
- VST3 SHA-256: `78942D244E44CB9F6BCA9197C3514E2A7073DBB528AF7652330673A6723337F4`.

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
Also pending: measured listening/reference comparisons and custom IR
import. The native Standard effect intentionally differs from Python; see the
plugin README for the source-history, interpolation, RNG and output-ceiling changes.
