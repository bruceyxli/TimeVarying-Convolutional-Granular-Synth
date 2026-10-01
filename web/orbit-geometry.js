/* Full parameter square <-> circular XY pad, shared by pointer and slider paths. */
(function(root) {
  const clamp = x => Math.max(-1, Math.min(1, x));
  const squareToDisc = (x, y) => { x = clamp(x); y = clamp(y); return [x * Math.sqrt(1 - y*y/2), y * Math.sqrt(1 - x*x/2)]; };
  const discToSquare = (x, y) => {
    const r = Math.hypot(x, y); if (r > 1) { x /= r; y /= r; }
    const a = 2 + x*x - y*y, b = 2 - x*x + y*y, k = 2 * Math.SQRT2, sqrt = v => Math.sqrt(Math.max(0, v));
    return [clamp((sqrt(a+k*x)-sqrt(a-k*x))/2), clamp((sqrt(b+k*y)-sqrt(b-k*y))/2)];
  };
  const api = {squareToDisc, discToSquare};
  if (typeof module !== "undefined") module.exports = api; else root.OrbitGeometry = api;
})(typeof window === "undefined" ? this : window);
