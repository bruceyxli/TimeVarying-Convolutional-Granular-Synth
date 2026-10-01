import io
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from src.app.audio_io import decode_audio
from src.app.engine import (
    GranularConfig, _IRConvolver, _pitch_factors, _pitch_resample,
    _resample_filter, render_offline,
)
from src.app.features import compute_spectral_centroid
from src.app.ir_bank import IRItem, IRSelector, generate_demo_ir_bank, select_ir_index


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = 48000
        cls.source = (0.1 * np.sin(2 * np.pi * 440 * np.arange(48000) / cls.sr)).astype(np.float32)
        cls.bank = generate_demo_ir_bank(num_irs=8)
        cls.cfg = GranularConfig(duration_sec=0.15, normalize=False)

    def test_variants_and_strategies_are_finite_and_reproducible(self):
        for variant in ("standard", "variant_a", "variant_b"):
            for strategy in sorted(IRSelector.STRATEGIES):
                with self.subTest(variant=variant, strategy=strategy):
                    cfg = replace(self.cfg, variant=variant, ir_strategy=strategy)
                    args = (self.source, self.bank, cfg, np.arange(1, 9))
                    actual = render_offline(*args)
                    np.testing.assert_array_equal(actual, render_offline(*args))
                    self.assertEqual(actual.shape, (7200, 2))
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertTrue(np.isfinite(actual).all())

    def test_cached_convolution_matches_scipy_including_tail(self):
        rng = np.random.default_rng(3)
        grain = rng.normal(size=257).astype(np.float32)
        items = [IRItem(rng.normal(size=n).astype(np.float32), 0) for n in (1, 127, 4097)]
        conv = _IRConvolver(items, len(grain))
        for index, item in enumerate(items):
            for _ in range(2):
                np.testing.assert_allclose(
                    conv.convolve(grain, index), fftconvolve(grain, item.samples),
                    atol=2e-5, rtol=2e-5,
                )
        self.assertEqual(len(conv.spectra), 3)

    def test_single_grain_matches_known_pan_mix_and_clipped_tail(self):
        cfg = replace(self.cfg, duration_sec=.01, density_hz=100, grain_ms=5,
                      pitch_semitones=0, jitter=0, pan_spread=0, wet=.7, dry=.3)
        impulse = np.zeros(400, dtype=np.float32)
        impulse[[0, 200, 399]] = [1, .5, -.2]
        grain = self.source[:240] * np.hanning(240).astype(np.float32)
        mono = .7 * np.convolve(grain, impulse)
        mono[:240] += .3 * grain
        expected = np.column_stack([mono[:480], mono[:480]]) / np.sqrt(2)
        actual = render_offline(self.source, [IRItem(impulse, 0)], cfg)
        np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_pitch_octaves_have_correct_frequency_and_non_silent_tail(self):
        for semi, expected in ((-12, 220), (0, 440), (12, 880)):
            with self.subTest(semitones=semi):
                result = _pitch_resample(self.source, semi, 4800)
                self.assertEqual(len(result), 4800)
                self.assertGreater(np.sqrt(np.mean(result[-1200:] ** 2)), 0.05)
                spectrum = np.abs(np.fft.rfft(result * np.hanning(len(result))))
                frequency = np.argmax(spectrum) * self.sr / len(result)
                self.assertAlmostEqual(frequency, expected, delta=10)

    def test_cached_resampler_matches_scipy_default(self):
        for semi in (-12, -7, -0.5, 0, 0.5, 7, 12, 24):
            up, down = _pitch_factors(semi)
            length = max(4, int(np.ceil(960 * down / up)))
            expected = resample_poly(self.source[:length], up, down)[:960]
            np.testing.assert_allclose(_pitch_resample(self.source, semi, 960), expected, atol=1e-7)
        self.assertLessEqual(_resample_filter.cache_info().currsize, 384)

    def test_variant_a_wet_and_dry_endpoints_and_linearity(self):
        cfg = replace(self.cfg, variant="variant_a")
        dry = render_offline(self.source, [], replace(cfg, wet=0, dry=1))
        wet = render_offline(self.source, [], replace(cfg, wet=1, dry=0))
        mixed = render_offline(self.source, [], replace(cfg, wet=0.6, dry=0.4))
        reference = render_offline(self.source, [], replace(cfg, variant="standard", wet=0, dry=1))
        np.testing.assert_allclose(dry, reference, atol=1e-6)
        self.assertGreater(np.max(np.abs(wet)), 0.01)
        np.testing.assert_allclose(mixed, 0.4 * dry + 0.6 * wet, atol=2e-6)

    def test_standard_mix_keeps_random_grain_sequence(self):
        for strategy in IRSelector.STRATEGIES:
            cfg = replace(self.cfg, ir_strategy=strategy)
            dry = render_offline(self.source, self.bank, replace(cfg, wet=0, dry=1))
            wet = render_offline(self.source, self.bank, replace(cfg, wet=1, dry=0))
            mixed = render_offline(self.source, self.bank, replace(cfg, wet=0.6, dry=0.4))
            np.testing.assert_allclose(mixed, 0.4 * dry + 0.6 * wet, atol=2e-6)

    def test_non_centroid_selection_skips_spectral_analysis(self):
        with patch("src.app.engine.compute_spectral_centroid", side_effect=AssertionError):
            render_offline(self.source, self.bank, replace(self.cfg, ir_strategy="fixed"))

    def test_zero_wet_skips_convolution(self):
        with patch("src.app.engine._IRConvolver.convolve", side_effect=AssertionError):
            render_offline(self.source, self.bank, replace(self.cfg, wet=0))

    def test_source_and_ir_arrays_are_not_modified(self):
        original = self.source.copy()
        bank = [item.samples.copy() for item in self.bank]
        for variant in ("standard", "variant_a", "variant_b"):
            render_offline(self.source, self.bank, replace(self.cfg, variant=variant))
        np.testing.assert_array_equal(self.source, original)
        for item, before in zip(self.bank, bank):
            np.testing.assert_array_equal(item.samples, before)

    def test_silence_short_input_and_output_bounds(self):
        for sr in (44100, 48000, 96000):
            for variant in ("standard", "variant_a", "variant_b"):
                cfg = replace(self.cfg, sample_rate=sr, duration_sec=.1, variant=variant)
                audio = render_offline(np.zeros(1, np.float32), self.bank, cfg)
                self.assertEqual(audio.shape, (int(sr * .1), 2))
                self.assertEqual(np.count_nonzero(audio), 0)
        loud = render_offline(self.source * 100, self.bank, replace(self.cfg, normalize=True))
        self.assertLessEqual(float(np.max(np.abs(loud))), 0.990001)
        empty_schedule = render_offline(self.source, self.bank, replace(self.cfg, duration_sec=.001))
        self.assertEqual(np.count_nonzero(empty_schedule), 0)

    def test_stereo_downmix(self):
        stereo = np.column_stack([self.source, -self.source])
        result = render_offline(stereo, self.bank, self.cfg)
        self.assertEqual(np.count_nonzero(result), 0)

    def test_invalid_config_fails_before_output_allocation(self):
        bad = {"sample_rate": [0, 48000.5], "duration_sec": [0, -1, np.nan, np.inf, 1e10],
               "density_hz": [0, -1, np.nan, 1e10], "grain_ms": [0, np.nan, 2000],
               "wet": [-1, np.nan, 2], "dry": [-1, 2], "jitter": [-1, 1],
               "pan_spread": [2], "pitch_semitones": [np.inf, -1, 25],
               "rng_seed": [-1, 1.5], "variant": ["bad"], "ir_strategy": ["bad"]}
        for field, values in bad.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    render_offline(self.source, self.bank, replace(self.cfg, **{field: value}))

    def test_invalid_sources_and_banks_fail_clearly(self):
        for source in ([], [np.nan], [np.inf], np.zeros((4, 0)), np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                render_offline(np.asarray(source), self.bank, self.cfg)
        for bank in ([], [IRItem(np.array([]), 0)], [IRItem(np.array([np.nan]), 0)],
                     [IRItem(np.ones(2), np.nan)]):
            with self.assertRaises(ValueError):
                render_offline(self.source, bank, self.cfg)

    def test_weight_validation_and_cycle_order(self):
        for weights in ([1], [-1] * 8, [0] * 8, [np.nan] * 8, [[1] * 8], [-1] + [1] * 7):
            with self.assertRaises(ValueError):
                IRSelector(self.bank, "weighted", np.asarray(weights))
        selector = IRSelector(self.bank, "weighted", np.full(8, 1e308))
        self.assertAlmostEqual(selector.weights.sum(), 1)
        rng = np.random.default_rng(5)
        counter = [-1]
        sequence = [select_ir_index("cycle", rng, self.bank, cycle_counter=counter) for _ in range(10)]
        self.assertEqual(sequence, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])

    def test_centroid_invalid_and_silent_frames(self):
        for signal in ([], [0], [np.nan, 1], [np.inf, 1]):
            self.assertEqual(compute_spectral_centroid(np.asarray(signal), self.sr), 0)
        self.assertAlmostEqual(compute_spectral_centroid(self.source, self.sr), 440, delta=5)

    def test_upload_decode_is_repeatable_and_resamples(self):
        file = io.BytesIO()
        sf.write(file, np.column_stack([self.source, self.source]), self.sr, format="WAV")
        data = file.getvalue()
        result = decode_audio(data, 44100)
        self.assertEqual(result.shape, (44100,))
        np.testing.assert_array_equal(result, decode_audio(data, 44100))
        with self.assertRaises((RuntimeError, ValueError)):
            decode_audio(b"broken audio", self.sr)


if __name__ == "__main__":
    unittest.main()
