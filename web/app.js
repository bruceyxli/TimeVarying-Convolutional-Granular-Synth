"use strict";
const $ = (id) => document.getElementById(id);
const controls = ["density", "grain", "pitch", "wet", "jitter", "pan", "seed", "variant", "strategy", "ir-ms", "bank-size", "long-ir", "duration", "sample-rate"];
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

function drawOrbit() {
  const { context: ctx, width: w, height: h } = fitCanvas($("orbit-art"));
  if (!w || !h) return;
  const x = (value("density") - 10) / 110, y = value("pitch") / 12;
  const cx = w / 2, cy = h / 2, s = Math.min(w, h);
  ctx.strokeStyle = "#293f56"; ctx.lineWidth = .6;
  ctx.beginPath(); ctx.arc(cx, cy, s * .414, 0, Math.PI * 2); ctx.stroke();
  for (let i = 0; i < 80; i++) {
    const a = i * Math.PI / 40;
    const inner = s * (i % 5 === 0 ? .425 : .431), outer = s * .438;
    ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
    ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer); ctx.stroke();
  }
  const glow = ctx.createRadialGradient(cx, cy, s * .16, cx, cy, s * .4);
  glow.addColorStop(0, "#82cfff00"); glow.addColorStop(.5, "#4c91d50d"); glow.addColorStop(1, "#82cfff00");
  ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);
  // Parameter-driven line sculpture. No video, bitmap asset or animation loop.
  const rings = 68;
  for (let line = 0; line < rings; line++) {
    const t = line / (rings - 1), phase = t * Math.PI * 2;
    const radius = s * (.176 + .18 * t);
    ctx.beginPath();
    for (let j = 0; j <= 200; j++) {
      const a = j / 200 * Math.PI * 2;
      const wave = (Math.sin(a * 3 + phase * 1.7 + x * 3) * .017
        + Math.sin(a * 5 - phase + y * 4) * .011) * s * (.5 + y);
      const twist = a + Math.sin(phase + a * 2) * (.04 + x * .08);
      const px = cx + Math.cos(twist) * (radius + wave) + Math.sin(phase) * s * .012;
      const py = cy + Math.sin(twist) * (radius + wave) * (.95 + .03 * Math.cos(phase));
      if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath(); ctx.lineWidth = .65;
    const alpha = .18 + .38 * Math.sin(t * Math.PI);
    ctx.strokeStyle = line % 6 === 0 ? `rgba(174,232,244,${alpha * .75})` : `rgba(123,184,244,${alpha})`;
    ctx.stroke();
  }
  ctx.strokeStyle = "#607f9f"; ctx.lineWidth = .6;
  ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy); ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
  $("xy-handle").style.left = `${(0.2 + x * .6) * 100}%`;
  $("xy-handle").style.top = `${(0.8 - y * .6) * 100}%`;
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
  return { source_id: state.sourceId, ir_ms: value("ir-ms"), num_irs: value("bank-size"), config: {
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
  $("density").value = Math.round(10 + x * 110); $("pitch").value = Math.round(y * 12);
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
