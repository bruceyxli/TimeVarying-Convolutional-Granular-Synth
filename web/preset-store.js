/* Parameter-only browser library; deliberately independent of audio/source data. */
(function (root) {
  "use strict";
  const key = "orbit.user-presets.v1";
  const schema = {
    density: [10, 120, 1], grain: [5, 50, .1], pitch: [0, 12, .1], wet: [0, 1, .001], reverb: [0, 1, .001],
    jitter: [0, .5, .01], pan: [0, 1, .01], seed: [0, 2147483647, 1],
    variant: ["standard", "variant_a", "variant_b"], strategy: ["weighted", "centroid", "random", "cycle", "fixed"],
    "ir-ms": [6, 32, 1], "bank-size": [8, 128, 1], "long-ir": [40, 300, 5], duration: [2, 40, .1], "sample-rate": [44100, 48000]
  };
  function valid(values) {
    if (!values || typeof values !== "object" || Array.isArray(values) || Object.keys(values).length !== Object.keys(schema).length) return false;
    return Object.entries(schema).every(([id, bounds]) => {
      const value = values[id];
      if (typeof bounds[0] === "string" || id === "sample-rate") return bounds.includes(value);
      return typeof value === "number" && Number.isFinite(value) && value >= bounds[0] && value <= bounds[1]
        && Math.abs((value - bounds[0]) / bounds[2] - Math.round((value - bounds[0]) / bounds[2])) < .00001;
    });
  }
  class Store {
    constructor(storage) { this.storage = storage; }
    list() {
      let data;
      try {
        const raw = this.storage.getItem(key);
        if (raw === null) return [];
        data = JSON.parse(raw);
      } catch { throw new Error("Could not read saved presets in this browser."); }
      if (!data || data.version !== 1 || !Array.isArray(data.presets)) throw new Error("Saved preset library has an unsupported format.");
      const seen = new Set();
      return data.presets.filter(p => {
        if (!p || typeof p.id !== "string" || !p.id || seen.has(p.id) || typeof p.name !== "string" || !p.name.trim() || p.name.length > 80 || !valid(p.values)) return false;
        seen.add(p.id); return true;
      }).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    }
    save(name, values) {
      name = String(name).trim();
      if (!name || name.length > 64 || /[\x00-\x1f]/.test(name)) throw new Error("Use a name with 1–64 characters.");
      if (!valid(values)) throw new Error("Check the parameter values before saving.");
      const presets = this.list();
      if (presets.length >= 256) throw new Error("The user preset library is full (256 presets).");
      const names = new Set(presets.map(p => p.name.toLowerCase()));
      let unique = name, suffix = 2;
      while (names.has(unique.toLowerCase())) unique = `${name} (${suffix++})`;
      const preset = {
        id: root.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        name: unique, values: JSON.parse(JSON.stringify(values))
      };
      presets.push(preset);
      try { this.storage.setItem(key, JSON.stringify({ version: 1, presets })); }
      catch { throw new Error("Could not save. Browser storage is full or unavailable."); }
      return preset;
    }
  }
  const api = { key, schema, valid, Store };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OrbitPresets = api;
})(globalThis);
