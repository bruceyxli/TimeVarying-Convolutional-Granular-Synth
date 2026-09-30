"use strict";
const $ = (id) => document.getElementById(id);
const controls = ["density", "grain", "pitch", "wet", "reverb", "jitter", "pan", "seed", "variant", "strategy", "ir-ms", "bank-size", "long-ir", "duration", "sample-rate"];
const state = { presets: {}, names: [], sourceId: null, sourceWave: [], demo: null, result: null, revision: 0, busy: false, uploading: false, online: false };
const value = (id) => Number($(id).value);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const percent = (x) => Math.round(x * 100);
const timecode = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong. Please try again.");
  return data;
}

function fitCanvas(canvas) {
  const { width, height } = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(width * ratio), h = Math.round(height * ratio);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawWave(canvas, peaks, progress = 0, colour = "#8cb8d6") {
  const { context: ctx, width: w, height: h } = fitCanvas(canvas);
  if (!w || !h) return;
  if (!peaks.length) {
    ctx.strokeStyle = "#2a3d52"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke(); return;
  }
  const maximum = Math.max(...peaks, .01);
  const step = w / peaks.length;
  ctx.lineWidth = Math.max(1, step * .5);
  peaks.forEach((peak, i) => {
    const size = Math.max(.7, peak / maximum * h * .38);
    ctx.strokeStyle = i / peaks.length < progress ? "#bbebff" : colour;
    ctx.beginPath(); ctx.moveTo(i * step, h / 2 - size); ctx.lineTo(i * step, h / 2 + size); ctx.stroke();
  });
}

let orbitVisual = null;
const orbitTargets = () => [(value("density") - 10) / 110, (value("grain") - 5) / 45, value("pitch") / 12, value("wet"), value("reverb")];
function drawOrbit() {
  const { context: ctx, width: w, height: h } = fitCanvas($("orbit-art"));
  if (!w || !h) return;
  const target = orbitTargets();
  if (!orbitVisual) orbitVisual = target.slice();
  let settling = false;
  orbitVisual = orbitVisual.map((v, i) => {
    const next = reducedMotion.matches ? target[i] : v + (target[i] - v) * .3;
    if (Math.abs(next - target[i]) < .001) return target[i];
    settling = true; return next;
  });
  const [x, grain, y, wet, space] = orbitVisual;
  const tint = Math.sqrt(space), rgb = [255 - 150 * tint, 255 - 64 * tint, 255].map(Math.round);
  const colour = (alpha) => `rgba(${rgb.join(",")},${alpha})`;
  $("orbit-pad").style.setProperty("--halo-colour", colour(1));
  $("orbit-pad").style.setProperty("--halo-glow", colour(space * .4));
  const cx = w / 2, cy = h / 2, s = Math.min(w, h);
  ctx.strokeStyle = "#293f56"; ctx.lineWidth = .6;
  ctx.beginPath(); ctx.arc(cx, cy, s * .414, 0, Math.PI * 2); ctx.stroke();
  for (let i = 0; i < 80; i++) {
    const a = i * Math.PI / 40;
    const inner = s * (i % 5 === 0 ? .425 : .431), outer = s * .438;
    ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
    ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer); ctx.stroke();
  }
  if (space > 0) {
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, s * .46);
    glow.addColorStop(0, colour(0)); glow.addColorStop(.48, colour(space * .025));
    glow.addColorStop(.69, colour(space * .14)); glow.addColorStop(.86, colour(space * .025)); glow.addColorStop(1, colour(0));
    ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);
  }
  // Animate only parameter transitions; no perpetual idle animation or fake audio.
  const rings = 28 + x * 36;
  for (let line = 0; line < Math.ceil(rings); line++) {
    const visibility = clamp(rings - line, 0, 1);
    const t = line / (rings - 1), phase = t * Math.PI * 2;
    const radius = s * (.272 + (t - .5) * (.085 + .13 * grain));
    ctx.beginPath();
    for (let j = 0; j <= 160; j++) {
      const a = j / 160 * Math.PI * 2;
      const wave = (Math.sin(a * 3 + phase * 1.4 + x * 2) * .019
        + Math.sin(a * 5 - phase + y * 3) * .012) * s * (.25 + y * 1.3);
      const twist = a + Math.sin(phase + a * 2) * (.035 + x * .075);
      const px = cx + Math.cos(twist) * (radius + wave) + Math.sin(phase) * s * .012;
      const py = cy + Math.sin(twist) * (radius + wave) * (.95 + .03 * Math.cos(phase));
      if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    const alpha = (.20 + .48 * Math.sin(t * Math.PI)) * (.65 + .35 * wet) * visibility;
    if (space > 0 && line % 5 === 0) {
      ctx.strokeStyle = colour(alpha * space * .055); ctx.lineWidth = 12 + space * 8; ctx.stroke();
      ctx.strokeStyle = colour(alpha * space * .14); ctx.lineWidth = 3; ctx.stroke();
    }
    ctx.lineWidth = .65 + .25 * wet; ctx.strokeStyle = colour(alpha); ctx.stroke();
  }
  ctx.strokeStyle = "#607f9f"; ctx.lineWidth = .6;
  ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy); ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
  $("xy-handle").style.left = `${(0.2 + target[0] * .6) * 100}%`;
  $("xy-handle").style.top = `${(0.8 - target[2] * .6) * 100}%`;
  if (settling) scheduleDraw();
}

let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; drawOrbit(); });
}

function updateUI(changed = true) {
  $("density-value").textContent = $("density").value;
  $("pitch-value").textContent = $("pitch").value;
  $("grain-value").textContent = $("grain").value;
  const room = value("reverb");
  $("reverb-value").textContent = room === 0 ? "OFF" : `${percent(room)}%`;
  $("reverb").setAttribute("aria-valuetext", room === 0 ? "Off" : `${percent(room)}%`);
  $("reverb-knob").style.setProperty("--amount", `${room * 270}deg`);
  $("reverb-knob").style.setProperty("--rotation", `${-135 + room * 270}deg`);
  $("reverb-knob").style.setProperty("--glow-size", `${room * 18}px`);
  document.querySelector(".reverb-control").style.setProperty("--halo-colour", room > 0 ? "#82cfff" : "#fff");
  $("wet").setAttribute("aria-valuetext", `${percent(value("wet"))}% wet`);
  $("wet-dial-value").textContent = percent(value("wet"));
  $("mix-dial").style.setProperty("--angle", `${value("wet") * 270}deg`);
  $("jitter-value").textContent = `${percent(value("jitter"))}%`;
  $("pan-value").textContent = `${percent(value("pan"))}%`;
  $("duration-value").textContent = `${value("duration")} s`;
  if (!state.result) $("total-time").textContent = ` / ${timecode(value("duration"))}`;
  document.querySelectorAll("input[type=range]").forEach((input) => {
    input.style.setProperty("--fill", `${(Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min)) * 100}%`);
  });
  const variant = $("variant").value;
  $("micro-ir-controls").hidden = variant !== "standard";
  $("long-ir-controls").hidden = variant !== "variant_a";
  if (changed) {
    state.revision++;
    if (state.result && !state.busy) message("Modified · render to update");
  }
  scheduleDraw();
}

function applyPreset() {
  const p = state.presets[$("preset").value];
  if (!p) return;
  const values = { density: p.density_hz, grain: p.grain_ms, jitter: p.jitter, pitch: p.pitch_semitones,
    pan: p.pan_spread, "ir-ms": p.ir_ms, strategy: p.ir_strategy, wet: p.wet,
    "bank-size": 64, "long-ir": 120, seed: 2025, variant: "standard" };
  Object.entries(values).forEach(([id, v]) => { $(id).value = v; });
  $("preset").title = p.description;
  updateUI();
}

function cyclePreset(direction) {
  if (!state.names.length) return;
  const index = (state.names.indexOf($("preset").value) + direction + state.names.length) % state.names.length;
  $("preset").value = state.names[index]; applyPreset();
}

function setSource(name, info, isDemo) {
  state.sourceId = info.source_id || null;
  state.sourceWave = info.waveform;
  $("source-name").textContent = name;
  $("source-name").title = name;
  $("source-kind").textContent = isDemo ? "DEMO" : "FILE";
  $("source-meta").textContent = `${info.duration.toFixed(1)} s`;
  $("use-demo").hidden = isDemo;
  $("duration").value = Math.round(clamp(info.duration, 2, 40) * 10) / 10;
  drawWave($("source-wave"), state.sourceWave);
  updateUI();
}

async function upload(file) {
  if (!file || state.uploading) return;
  if (file.size > 24 * 1024 * 1024) return message("Choose an audio file smaller than 24 MB.", true);
  state.uploading = true; setButtons(); message("Importing…");
  try {
    const data = await api("/api/source", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file });
    setSource(file.name.replace(/\.[^.]+$/, ""), data, false);
    message(state.result ? "Source changed · render to update" : "");
  } catch (error) { message(error.message, true); }
  finally { state.uploading = false; $("source-file").value = ""; setButtons(); }
}

function settings() {
  return { source_id: state.sourceId, ir_ms: value("ir-ms"), num_irs: value("bank-size"), reverb: value("reverb"), config: {
    sample_rate: value("sample-rate"), duration_sec: value("duration"), density_hz: value("density"),
    grain_ms: value("grain"), pitch_semitones: value("pitch"), jitter: value("jitter"), pan_spread: value("pan"),
    wet: value("wet"), dry: 1 - value("wet"), ir_strategy: $("strategy").value,
    variant: $("variant").value, long_ir_ms: value("long-ir"), rng_seed: value("seed"), normalize: true
  } };
}

function setButtons() {
  $("render").disabled = !state.online || state.busy || state.uploading;
  $("upload-button").disabled = !state.online || state.uploading || state.busy;
  $("render").classList.toggle("busy", state.busy && !reducedMotion.matches);
  $("render-label").textContent = state.busy ? "Rendering…" : "Render";
}

async function render() {
  for (const id of controls) {
    if (!$(id).checkValidity()) {
      $("detail-panel").hidden = false; $("detail-toggle").setAttribute("aria-expanded", "true");
      $(id).reportValidity(); return;
    }
  }
  const revision = state.revision;
  const request = settings();
  state.busy = true; setButtons(); message("");
  try {
    const result = await api("/api/render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    state.result = result;
    $("audio").pause(); $("audio").src = result.url;
    $("seek").value = 0; $("seek").disabled = false;
    $("play").disabled = false;
    $("current-time").textContent = "0:00";
    $("total-time").textContent = ` / ${timecode(result.duration)}`;
    $("download").href = result.url;
    $("download").download = `ORBIT-${$("preset").value.replace(/[^a-z0-9 -]/gi, "")}-${Date.now()}.wav`;
    $("download").setAttribute("aria-disabled", "false");
    $("download").title = `24-bit stereo WAV · ${result.sample_rate} Hz · ${result.peak_db} dBFS peak`;
    drawWave($("output-wave"), result.waveform);
    message(state.revision === revision ? "Ready" : "Modified · render to update");
  } catch (error) { message(error.message, true); }
  finally { state.busy = false; setButtons(); }
}

controls.forEach((id) => $(id).addEventListener("input", () => updateUI()));
// Keep real range inputs for keyboard/assistive control, add fine pointer movement.
["density", "grain", "pitch", "wet", "reverb"].forEach((id) => {
  const input = $(id), initial = input.defaultValue;
  let drag = null;
  const set = (v) => {
    const step = Number(input.step), lo = Number(input.min), hi = Number(input.max);
    input.value = clamp(Math.round((v - lo) / step) * step + lo, lo, hi).toFixed(3);
    updateUI();
  };
  input.title = "Shift-drag for precision · Double-click to reset";
  input.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !event.isPrimary) return;
    event.preventDefault(); input.focus(); input.setPointerCapture(event.pointerId);
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, value: Number(input.value) };
    if (id !== "reverb" && !event.shiftKey) {
      const bounds = input.getBoundingClientRect();
      set(Number(input.min) + clamp((event.clientX - bounds.left - 8) / (bounds.width - 16), 0, 1) * (Number(input.max) - Number(input.min)));
      drag.value = Number(input.value);
    }
  });
  input.addEventListener("pointermove", (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    const range = Number(input.max) - Number(input.min), bounds = input.getBoundingClientRect();
    if (id === "reverb" || event.shiftKey) {
      const delta = id === "reverb" ? (drag.y - event.clientY) + (event.clientX - drag.x) * .5 : event.clientX - drag.x;
      drag.value = clamp(drag.value + delta / (id === "reverb" ? 150 : bounds.width) * range * (event.shiftKey ? .1 : 1), Number(input.min), Number(input.max));
      set(drag.value);
    } else {
      set(Number(input.min) + clamp((event.clientX - bounds.left - 8) / (bounds.width - 16), 0, 1) * range);
      drag.value = Number(input.value);
    }
    drag.x = event.clientX; drag.y = event.clientY;
  });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => input.addEventListener(name, () => { drag = null; }));
  input.addEventListener("dblclick", () => set(Number(initial)));
});
$("preset").addEventListener("change", applyPreset);
$("previous-preset").addEventListener("click", () => cyclePreset(-1));
$("next-preset").addEventListener("click", () => cyclePreset(1));
$("reset").addEventListener("click", applyPreset);
$("detail-toggle").addEventListener("click", () => {
  const open = $("detail-panel").hidden;
  $("detail-panel").hidden = !open;
  $("detail-toggle").setAttribute("aria-expanded", String(open));
  $("detail-sign").textContent = open ? "−" : "+";
});
$("upload-button").addEventListener("click", () => $("source-file").click());
$("source-file").addEventListener("change", () => upload($("source-file").files[0]));
$("use-demo").addEventListener("click", () => { if (state.demo) setSource(state.demo.name, state.demo, true); });
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("drop-zone").classList.add("dragging"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragging"));
$("drop-zone").addEventListener("drop", (event) => {
  event.preventDefault(); $("drop-zone").classList.remove("dragging");
  if (!state.busy) upload(event.dataTransfer.files[0]);
});
$("render").addEventListener("click", render);
$("play").addEventListener("click", async () => {
  try { if ($("audio").paused) await $("audio").play(); else $("audio").pause(); }
  catch { message("Playback could not start. Try rendering again.", true); }
});
function playbackUI() {
  const playing = !$("audio").paused;
  $("play").textContent = playing ? "Ⅱ" : "▶";
  $("play").setAttribute("aria-label", playing ? "Pause rendered audio" : "Play rendered audio");
}
$("audio").addEventListener("play", playbackUI);
$("audio").addEventListener("pause", playbackUI);
$("audio").addEventListener("ended", playbackUI);
$("audio").addEventListener("error", () => {
  $("play").disabled = true;
  message("This render is no longer available. Render again to listen.", true);
});
$("audio").addEventListener("timeupdate", () => {
  const progress = $("audio").currentTime / (state.result?.duration || 1);
  $("seek").value = Math.round(progress * 1000);
  $("current-time").textContent = timecode($("audio").currentTime);
  drawWave($("output-wave"), state.result?.waveform || [], progress);
});
$("seek").addEventListener("input", () => { if (state.result) $("audio").currentTime = value("seek") / 1000 * state.result.duration; });

let pointer = null;
function moveXY(event) {
  const rect = $("orbit-pad").getBoundingClientRect();
  const x = clamp(((event.clientX - rect.left) / rect.width - .2) / .6, 0, 1);
  const y = clamp((.8 - (event.clientY - rect.top) / rect.height) / .6, 0, 1);
  $("density").value = Math.round(10 + x * 110); $("pitch").value = (y * 12).toFixed(1);
  updateUI();
}
$("orbit-pad").addEventListener("pointerdown", (event) => {
  if (!event.isPrimary || event.button !== 0) return;
  pointer = event.pointerId; $("orbit-pad").setPointerCapture(pointer); moveXY(event);
});
$("orbit-pad").addEventListener("pointermove", (event) => { if (event.pointerId === pointer) moveXY(event); });
$("orbit-pad").addEventListener("pointerup", () => { pointer = null; });
$("orbit-pad").addEventListener("pointercancel", () => { pointer = null; });
$("orbit-pad").addEventListener("lostpointercapture", () => { pointer = null; });
new ResizeObserver(() => {
  scheduleDraw(); drawWave($("source-wave"), state.sourceWave);
  drawWave($("output-wave"), state.result?.waveform || [], $("audio").currentTime / (state.result?.duration || 1));
}).observe(document.querySelector(".instrument"));

async function start() {
  updateUI(false);
  try {
    const data = await api("/api/bootstrap");
    state.presets = data.presets; state.names = Object.keys(data.presets); state.demo = data.source;
    $("preset").replaceChildren(...state.names.map((name) => new Option(name, name)));
    $("preset").value = "Airy shimmer";
    setSource(data.source.name, data.source, true); applyPreset();
    state.online = true;
    document.querySelector(".engine-status").setAttribute("aria-label", "Engine ready");
    document.querySelector(".engine-status").title = "Engine ready";
    $("engine-dot").classList.add("ready"); setButtons();
  } catch (error) {
    document.querySelector(".engine-status").setAttribute("aria-label", "Engine offline");
    document.querySelector(".engine-status").title = "Engine offline";
    message("Engine offline · restart the local server and reload", true);
  }
}
start();
