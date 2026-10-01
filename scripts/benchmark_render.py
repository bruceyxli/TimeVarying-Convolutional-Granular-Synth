"""Reproducible, subprocess-isolated offline benchmarks (not callback timings).

python scripts/benchmark_render.py --compare-ref <git-ref> --output report.json
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path


def measure(root: Path, repeats: int):
    sys.path.insert(0, str(root))
    import numpy as np
    import scipy
    from src.app import engine
    from src.app.ir_bank import generate_demo_ir_bank

    scenarios = {
        "default_10s": {},
        "dense_40s": {"duration_sec": 40, "density_hz": 120, "grain_ms": 50},
        "centroid_10s": {"ir_strategy": "centroid"},
        "variant_a_10s": {"variant": "variant_a"},
        "variant_b_10s": {"variant": "variant_b"},
        "zero_pitch_10s": {"pitch_semitones": 0},
    }
    results = {}
    for name, params in scenarios.items():
        cfg = engine.GranularConfig(**params)
        sr = cfg.sample_rate
        t = np.arange(int(sr * cfg.duration_sec)) / sr
        source = (0.1 * np.sin(2 * np.pi * 220 * t)
                  + 0.05 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        bank = generate_demo_ir_bank(sample_rate=sr)
        weights = np.linspace(1.0, 2.0, len(bank))
        if hasattr(engine, "_resample_filter"):
            engine._resample_filter.cache_clear()
        begin = time.perf_counter()
        audio = engine.render_offline(source, bank, cfg, weights)
        cold = time.perf_counter() - begin
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Non-finite output: {name}")
        del audio
        elapsed = []
        for _ in range(repeats):
            begin = time.perf_counter()
            audio = engine.render_offline(source, bank, cfg, weights)
            elapsed.append(time.perf_counter() - begin)
            del audio
        # Separate run: tracing affects timing, so exclude it from speed results.
        tracemalloc.start()
        engine.render_offline(source, bank, cfg, weights)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        median = statistics.median(elapsed)
        results[name] = {
            "cold_seconds": cold,
            "median_seconds": median,
            "max_seconds": max(elapsed),
            "playback_speed_multiple": cfg.duration_sec / median,
            "traced_peak_mib": peak / 1024 ** 2,
        }
    return {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "platform": platform.platform(),
                            "processor": platform.processor()},
            "repeats": repeats, "scenarios": results}


def run_worker(root, repeats):
    output = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--root", str(root),
         "--repeats", str(repeats)], text=True,
    )
    return json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-ref")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        print(json.dumps(measure(args.root, args.repeats)))
        return
    report = {}
    if args.compare_ref:
        ref = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.compare_ref + "^{commit}"],
            cwd=args.root, text=True,
        ).strip()
        # Extract only required source modules; never switch the working checkout.
        with tempfile.TemporaryDirectory(prefix="granular-baseline-") as directory:
            baseline = Path(directory)
            for name in ("__init__.py", "engine.py", "features.py", "ir_bank.py"):
                relative = f"src/app/{name}"
                data = subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=args.root)
                target = baseline / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            report["baseline_ref"] = ref
            report["baseline"] = run_worker(baseline, args.repeats)
    report["current"] = run_worker(args.root, args.repeats)
    if "baseline" in report:
        report["speedup"] = {
            name: previous["median_seconds"] / report["current"]["scenarios"][name]["median_seconds"]
            for name, previous in report["baseline"]["scenarios"].items()
        }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
