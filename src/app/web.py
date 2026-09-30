"""Local, dependency-free HTTP adapter for the lightweight instrument page.

Run from the repository root: python -m src.app.web
The DSP remains in engine.py; this server is for local offline rendering only.
"""
import argparse
from collections import OrderedDict
from dataclasses import fields
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
import soundfile as sf

from src.app.audio_io import decode_audio
from src.app.engine import GranularConfig, render_offline
from src.app.ir_bank import generate_demo_ir_bank
from src.app.presets import PRESETS
from src.app.reverb import apply_reverb

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
MAX_UPLOAD = 24 * 1024 * 1024


def waveform(audio, bins=180):
    """Peak envelope from actual audio, bounded independently of clip length."""
    mono = np.max(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    return [round(float(np.max(part)), 5) for part in np.array_split(mono, min(bins, len(mono)))]


@lru_cache(maxsize=2)
def demo_source(sample_rate):
    t = np.arange(sample_rate * 10) / sample_rate
    rng = np.random.default_rng(2025)
    pad = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63)) / 3
    envelope = np.minimum(t / .5, 1) * (.65 + .35 * np.sin(2 * np.pi * .23 * t) ** 2)
    return (pad * envelope * .3 + rng.normal(0, .014, len(t))).astype(np.float32)


@lru_cache(maxsize=4)
def prepared_bank(count, milliseconds, sample_rate, seed):
    return generate_demo_ir_bank(count, milliseconds, sample_rate, seed)


class AudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, Handler)
        self.sources = OrderedDict()
        self.renders = OrderedDict()
        self.store_lock = threading.Lock()
        self.render_lock = threading.Lock()

    def store(self, collection, value):
        token = uuid4().hex
        with self.store_lock:
            collection[token] = value
            while len(collection) > 4:
                collection.popitem(last=False)
        return token

    def lookup(self, collection, token):
        with self.store_lock:
            if token not in collection:
                raise ValueError("This audio has expired. Reload the source or render again.")
            return collection[token]


def render_request(server, data):
    if not isinstance(data, dict):
        raise ValueError("Expected a settings object")
    settings = data.get("config", {})
    if not isinstance(settings, dict) or set(settings) - {f.name for f in fields(GranularConfig)}:
        raise ValueError("Unknown render settings")
    cfg = GranularConfig(**settings)
    cfg.validate()
    reverb = data.get("reverb", 0)
    if not isinstance(reverb, (int, float)) or not np.isfinite(reverb) or not 0 <= reverb <= 1:
        raise ValueError("Reverb must be between 0 and 1")
    if not 2 <= cfg.duration_sec <= 40 or cfg.density_hz > 120 or cfg.grain_ms > 50:
        raise ValueError("Web renders support 2–40 seconds, up to 120 grains/s and 50 ms grains")
    if cfg.sample_rate not in (44100, 48000) or cfg.long_ir_ms > 300:
        raise ValueError("Choose 44.1/48 kHz and a long IR up to 300 ms")
    count, milliseconds = data.get("num_irs", 64), data.get("ir_ms", 16)
    if (type(count) is not int or not 8 <= count <= 128
            or not isinstance(milliseconds, (int, float)) or not 6 <= milliseconds <= 32):
        raise ValueError("IR bank must contain 8–128 items of 6–32 ms")
    source_id = data.get("source_id")
    if source_id is not None and not isinstance(source_id, str):
        raise ValueError("Invalid source ID")
    source = decode_audio(server.lookup(server.sources, source_id), cfg.sample_rate) if source_id else demo_source(cfg.sample_rate)
    bank = prepared_bank(count, milliseconds, cfg.sample_rate, cfg.rng_seed) if cfg.variant == "standard" else []
    weights = np.linspace(1, 2, len(bank)) if cfg.ir_strategy == "weighted" else None
    start = perf_counter()
    audio = render_offline(source, bank, cfg, weights)
    audio = apply_reverb(audio, cfg.sample_rate, reverb)
    if reverb > 0:
        audio /= max(1.0, float(np.max(np.abs(audio))))
    elapsed = perf_counter() - start
    buffer = io.BytesIO()
    sf.write(buffer, audio, cfg.sample_rate, format="WAV", subtype="PCM_24")
    token = server.store(server.renders, buffer.getvalue())
    peak = float(np.max(np.abs(audio)))
    return {"url": f"/api/audio/{token}", "duration": cfg.duration_sec,
            "sample_rate": cfg.sample_rate, "render_seconds": elapsed,
            "peak_db": round(20 * np.log10(max(peak, 1e-9)), 1), "waveform": waveform(audio)}


class Handler(BaseHTTPRequestHandler):
    server: AudioServer

    def log_message(self, format, *args):
        pass

    def respond(self, content, mime="application/json", status=200):
        body = json.dumps(content, allow_nan=False).encode() if mime == "application/json" else content
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.valid_host():
            return self.respond({"error": "Local access only"}, status=403)
        path = urlsplit(self.path).path
        if path == "/api/bootstrap":
            return self.respond({"presets": PRESETS, "source": {"name": "Harmonic drift", "duration": 10,
                                 "waveform": waveform(demo_source(48000))}})
        if path.startswith("/api/audio/"):
            try:
                return self.respond(self.server.lookup(self.server.renders, path.rsplit("/", 1)[1]), "audio/wav")
            except ValueError as exc:
                return self.respond({"error": str(exc)}, status=404)
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/preset-store.js": ("preset-store.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/fonts/oxanium.ttf": ("fonts/oxanium.ttf", "font/ttf")}
        if path not in files:
            return self.respond({"error": "Not found"}, status=404)
        name, mime = files[path]
        self.respond((WEB_ROOT / name).read_bytes(), mime)

    def do_POST(self):
        if not self.valid_host() or self.headers.get("Origin") not in (None, "http://" + self.headers.get("Host", "")):
            return self.respond({"error": "Local same-origin requests only"}, status=403)
        path = urlsplit(self.path).path
        if path not in ("/api/source", "/api/render"):
            return self.respond({"error": "Not found"}, status=404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= (MAX_UPLOAD if path == "/api/source" else 16384):
                return self.respond({"error": "Upload limit is 24 MB"}, status=413)
            self.connection.settimeout(30)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Incomplete request")
            if path == "/api/source":
                info = sf.info(io.BytesIO(payload))
                if (info.frames <= 0 or info.duration > 120 or info.channels > 8
                        or info.frames * info.channels > 8_000_000):
                    raise ValueError("Use audio up to 120 seconds and 8 million decoded samples")
                source = decode_audio(payload, 48000)
                token = self.server.store(self.server.sources, payload)
                return self.respond({"source_id": token, "duration": info.duration,
                                     "waveform": waveform(source)})
            data = json.loads(payload)
            if not self.server.render_lock.acquire(blocking=False):
                return self.respond({"error": "A render is already running. Try again shortly."}, status=409)
            try:
                result = render_request(self.server, data)
            finally:
                self.server.render_lock.release()
            self.respond(result)
        except (ValueError, TypeError, OverflowError, RuntimeError, OSError) as exc:
            self.respond({"error": str(exc)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with AudioServer(("127.0.0.1", args.port)) as server:
        print(f"ORBIT is ready at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
