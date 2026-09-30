"use strict";
const $ = (id) => document.getElementById(id);
const controls = ["density", "grain", "pitch", "wet", "reverb", "jitter", "pan", "seed", "variant", "strategy", "ir-ms", "bank-size", "long-ir", "duration", "sample-rate"];
const state = { presets: {}, names: [], sourceId: null, sourceWave: [], demo: null, result: null, revision: 0, busy: false, uploading: false, online: false };
state.userPresets = []; state.loadedValues = null;
const presetLibrary = new OrbitPresets.Store({ getItem: key => localStorage.getItem(key), setItem: (key, text) => localStorage.setItem(key, text) });
const presetSnapshot = () => Object.fromEntries(controls.map(id => [id, ["variant", "strategy"].includes(id) ? $(id).value : Number($(id).value)]));
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
  updatePresetLabel();
}

function applyPreset() {
  const user = state.userPresets.find(p => `user:${p.id}` === $("preset").value);
  if (user) {
    Object.entries(user.values).forEach(([id, v]) => { $(id).value = v; });
    state.loadedValues = presetSnapshot();$("preset").title = user.name;updateUI();return;
  }
  const p = state.presets[$("preset").value];
  if (!p) return;
  const values = { density: p.density_hz, grain: p.grain_ms, jitter: p.jitter, pitch: p.pitch_semitones,
    pan: p.pan_spread, "ir-ms": p.ir_ms, strategy: p.ir_strategy, wet: p.wet,
    "bank-size": 64, "long-ir": 120, seed: 2025, variant: "standard", reverb: 0, duration: 10, "sample-rate": 48000 };
  Object.entries(values).forEach(([id, v]) => { $(id).value = v; });
  $("preset").title = p.description;
  state.loadedValues = presetSnapshot();
  updateUI();
}

function updatePresetLabel() {
  const selected = $("preset").selectedOptions[0];
  if (!selected || !state.loadedValues) return;
  const current = presetSnapshot();
  const modified = controls.some(id => current[id] !== state.loadedValues[id]);
  const name = selected.dataset.name || selected.value;
  selected.textContent = name + (modified ? " *" : "");
}
function refreshPresetOptions(selected = $("preset").value) {
  const factory = document.createElement("optgroup");factory.label = "Factory";
  Object.keys(state.presets).forEach(name => { const option = new Option(name, name);option.dataset.name = name;factory.append(option); });
  const user = document.createElement("optgroup");user.label = "User";
  state.userPresets.forEach(p => { const option = new Option(p.name, `user:${p.id}`);option.dataset.name = p.name;user.append(option); });
  $("preset").replaceChildren(factory, ...(state.userPresets.length ? [user] : []));
  state.names = [...Object.keys(state.presets), ...state.userPresets.map(p => `user:${p.id}`)];
  $("preset").value = state.names.includes(selected) ? selected : state.names[0];
  updatePresetLabel();
}
$("save-preset").addEventListener("click", () => {
  $("preset-save-error").textContent = "";
  $("preset-name").value = state.userPresets.find(p => `user:${p.id}` === $("preset").value)?.name || "My preset";
  $("save-preset-dialog").showModal();$("preset-name").focus();$("preset-name").select();
});
$("cancel-preset-save").addEventListener("click", () => $("save-preset-dialog").close());
$("save-preset-form").addEventListener("submit", event => {
  event.preventDefault();
  try {
    const saved = presetLibrary.save($("preset-name").value, presetSnapshot());
    state.userPresets = presetLibrary.list();state.loadedValues = presetSnapshot();
    refreshPresetOptions(`user:${saved.id}`);$("preset").title=saved.name;$("save-preset-dialog").close();message("Preset saved");
  } catch (error) { $("preset-save-error").textContent = error.message; }
});
window.addEventListener("storage", event => {
  if (event.key !== OrbitPresets.key && event.key !== null) return;
  try { state.userPresets = presetLibrary.list();refreshPresetOptions(); }
  catch (error) { message(error.message, true); }
});

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
      setDetailsPage(Boolean($(id).closest("#detail-panel")));
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
  // Parameter help is installed below for sliders, labels and readouts.
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
function setDetailsPage(open) {
  const instrument = document.querySelector(".instrument"), workspace = document.querySelector(".workspace");
  if (open && !workspace.hidden) instrument.style.setProperty("--details-height", `${workspace.offsetHeight}px`);
  workspace.hidden = open;
  document.querySelector(".detail-section").hidden = !open;
  instrument.classList.toggle("details-open", open);
  $("detail-panel").hidden = !open;
  $("details-title").hidden = !open;
  $("detail-toggle").setAttribute("aria-label", open ? "Back to instrument" : "Open Details");
  $("detail-toggle").innerHTML = open ? '<span aria-hidden="true">←</span> Back' : 'Details <span aria-hidden="true">→</span>';
  window.scrollTo({top:0,behavior:"instant"});
  if (open) $("details-title").focus({preventScroll:true});
  else { $("detail-toggle").focus({preventScroll:true}); scheduleDraw();drawWave($("source-wave"), state.sourceWave); }
}
$("detail-toggle").addEventListener("click", () => setDetailsPage($("detail-panel").hidden));
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
  pointer = event.pointerId; $("orbit-pad").setPointerCapture(pointer);
  $("orbit-pad").classList.add("is-dragging"); moveXY(event);
});
$("orbit-pad").addEventListener("pointermove", (event) => {
  if (event.pointerId === pointer) moveXY(event);
  const bounds = $("xy-handle").getBoundingClientRect();
  $("orbit-pad").classList.toggle("handle-hover", Math.hypot(event.clientX - bounds.left - bounds.width / 2, event.clientY - bounds.top - bounds.height / 2) < 22);
});
const releaseXY = () => { pointer = null; $("orbit-pad").classList.remove("is-dragging", "handle-hover"); };
["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => $("orbit-pad").addEventListener(name, releaseXY));
$("orbit-pad").addEventListener("pointerleave", () => $("orbit-pad").classList.remove("handle-hover"));
new ResizeObserver(() => {
  scheduleDraw(); drawWave($("source-wave"), state.sourceWave);
  drawWave($("output-wave"), state.result?.waveform || [], $("audio").currentTime / (state.result?.duration || 1));
}).observe(document.querySelector(".instrument"));

async function start() {
  updateUI(false);
  try {
    const data = await api("/api/bootstrap");
    state.presets = data.presets; state.demo = data.source;
    try { state.userPresets = presetLibrary.list(); } catch (error) { message(error.message, true); }
    refreshPresetOptions();
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

// One lightweight tooltip; descriptions are also available to screen readers.
function installParameterHelp() {
  const descriptions = {
    "density": "Density · 颗粒密度\n每秒触发的颗粒数。越高越密集，越低越稀疏。",
    "grain": "Grain size · 颗粒时长\n每个颗粒持续的时间。较短更细碎，较长保留更多原声细节。",
    "pitch": "Pitch scatter · 音高散布\n每个颗粒随机升降音高的范围，以半音计。0 保持原音高。",
    "wet": "Dry / Wet · 干湿比\n混合原始输入与颗粒效果。0% 为原声，100% 为颗粒声；Reverb 在混合后加入。",
    "reverb": "Reverb · 混响\n增加空间感与尾音。0 关闭混响，光环为白色；增大后光环变蓝、光晕增强。",
    "jitter": "Jitter · 触发抖动\n随机偏移颗粒触发时间。0 更规律，增大后节奏更松散。",
    "pan": "Spread · 立体声散布\n颗粒在左右声道间随机分布的宽度。0 居中，增大后更宽。",
    "seed": "Seed · 随机种子\n改变随机变化的序列。相同输入、种子与起始状态可复现相同变化。",
    "variant": "Signal path · 信号路径\n选择逐颗粒卷积、先卷积再颗粒化，或用颗粒作为脉冲响应。",
    "strategy": "Selection · IR 选择\n选择每个颗粒的响应：固定、轮换、随机、加权，或按频谱重心匹配。",
    "ir-ms": "IR · 脉冲响应长度\n控制每颗粒卷积的短响应时长。较短更紧凑，较长带来更多共鸣。",
    "bank-size": "Bank size · IR 数量\n生成的短脉冲响应数量。越多，可选音色越丰富，也会增加准备时间。",
    "long-ir": "Long IR · 长响应\n先卷积模式下的脉冲响应时长。越长，空间尾音越明显。",
    "duration": "Duration · 渲染时长\n生成音频的总长度，单位为秒。",
    "sample-rate": "Sample rate · 采样率\n选择输出音频的采样率。更高采样率会增加运算量与文件体积。",
    "orbit-pad": "XY · 声音控制\n左右改变颗粒密度，上下改变音高散布。也可使用两侧推子调整。"
  };

  const tooltip = document.createElement("div");
  tooltip.className = "parameter-tooltip"; tooltip.hidden = true;
  tooltip.setAttribute("aria-hidden", "true"); document.body.append(tooltip);
  let active = null, timer = 0, pointerHeld = false;
  function hide() { clearTimeout(timer); tooltip.hidden = true; active = null; }
  function show(target, delay = 550) {
    if (!target || pointerHeld || target === active) return;
    hide(); active = target;
    timer = setTimeout(() => {
      if (!target.isConnected || !target.getClientRects().length) return hide();
      const [heading, ...body] = descriptions[target.dataset.help].split("\n");
      const title = document.createElement("strong"); title.textContent = heading;
      const text = document.createElement("span"); text.textContent = body.join(" ");
      tooltip.replaceChildren(title, text); tooltip.hidden = false;
      const bounds = target.getBoundingClientRect(), gap = 10;
      const left = Math.max(gap, Math.min(bounds.left + bounds.width / 2 - tooltip.offsetWidth / 2, innerWidth - tooltip.offsetWidth - gap));
      let top = bounds.bottom + gap;
      if (top + tooltip.offsetHeight > innerHeight - gap) top = bounds.top - tooltip.offsetHeight - gap;
      tooltip.style.left = left + "px"; tooltip.style.top = Math.max(gap, top) + "px";
    }, delay);
  }
  Object.entries(descriptions).forEach(([id, description]) => {
    const input = $(id), accessible = document.createElement("span");
    accessible.id = "help-" + id; accessible.className = "sr-only"; accessible.textContent = description;
    document.body.append(accessible); input.setAttribute("aria-describedby", accessible.id);
    input.removeAttribute("title");
    const targets = [input, ...document.querySelectorAll('label[for="' + id + '"]')];
    const parameter = input.closest(".parameter");
    if (parameter) targets.push(parameter);
    if (id === "wet") targets.push(document.querySelector(".mix-section"));
    if (id === "reverb") targets.push(input.closest(".reverb-control"));
    targets.filter(Boolean).forEach(target => { target.dataset.help = id; });
  });
  const targetOf = element => element instanceof Element ? element.closest("[data-help]") : null;
  document.addEventListener("pointerover", event => {
    if (event.pointerType === "touch") return;
    const target = targetOf(event.target);
    if (target !== targetOf(event.relatedTarget)) show(target);
  });
  document.addEventListener("pointerout", event => {
    const next = targetOf(event.relatedTarget);
    if (targetOf(event.target) !== next) { hide(); if (next) show(next); }
  });
  document.addEventListener("focusin", event => show(targetOf(event.target), 350));
  document.addEventListener("focusout", hide);
  document.addEventListener("pointerdown", () => { pointerHeld = true; hide(); }, true);
  document.addEventListener("pointerup", () => { pointerHeld = false; }, true);
  document.addEventListener("pointercancel", () => { pointerHeld = false; hide(); }, true);
  document.addEventListener("keydown", event => { if (event.key === "Escape") hide(); });
  window.addEventListener("blur", () => { pointerHeld = false; hide(); });
  window.addEventListener("resize", hide);
  document.addEventListener("scroll", hide, true);
}
installParameterHelp();
