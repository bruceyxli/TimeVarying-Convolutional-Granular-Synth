import unittest
import numpy as np
from src.app.reverb import apply_reverb, room_impulses


class ReverbTests(unittest.TestCase):
    def test_zero_is_exact_and_does_not_modify_input(self):
        audio = np.random.default_rng(2).normal(0, .1, (1000, 2)).astype(np.float32)
        saved = audio.copy()
        np.testing.assert_array_equal(apply_reverb(audio, 48000, 0), audio)
        wet = apply_reverb(audio, 48000, 1)
        np.testing.assert_array_equal(audio, saved)
        self.assertTrue(np.isfinite(wet).all())

    def test_room_has_stereo_tail_and_decays(self):
        sr = 8000
        audio = np.zeros((sr * 3, 2), dtype=np.float32)
        audio[0, 0] = .4
        wet = apply_reverb(audio, sr, 1)
        self.assertGreater(float(np.sum(np.abs(wet[200:4000, 0]))), .01)
        self.assertGreater(float(np.sum(np.abs(wet[200:4000, 1]))), .001)
        self.assertLess(float(np.sum(np.abs(wet[-4000:]))), .001)
        np.testing.assert_allclose(wet[0], [.28, 0], atol=1e-7)

    def test_prepared_kernel_matches_sample_recurrence(self):
        # Independent sample-by-sample implementation of the native room equations.
        sr, count = 8000, 2400
        for channel, kernel in enumerate(room_impulses(sr)):
            impulse = np.zeros(count);impulse[0] = 1
            room = np.zeros(count)
            for seconds in (.0297, .0371, .0411, .0437):
                delay = int(sr * (seconds + channel * .00079))
                data = np.zeros(delay);head = 0;damped = 0
                for i in range(count):
                    out = data[head];damped = .75 * out + .25 * damped
                    data[head] = impulse[i] + .76 * damped
                    head = (head + 1) % delay;room[i] += out * .25
            for seconds, offset in ((.005, .00041), (.0017, .00019)):
                delay = int(sr * (seconds + channel * offset));data = np.zeros(delay);head = 0
                for i in range(count):
                    sample = room[i];room[i] = data[head] - .5 * sample
                    data[head] = sample + .5 * room[i];head = (head + 1) % delay
            np.testing.assert_allclose(kernel[:count], room, atol=1e-7)

    def test_invalid_amount_rejected(self):
        for amount in (-1, 2, np.nan, np.inf):
            with self.assertRaises(ValueError):
                apply_reverb(np.zeros((10, 2)), 48000, amount)
