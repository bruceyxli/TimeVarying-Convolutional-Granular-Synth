"""Offline impulse-response version of the native damped stereo room."""
from functools import lru_cache
import numpy as np
from scipy.signal import fftconvolve, lfilter


@lru_cache(maxsize=2)
def room_impulses(sample_rate):
    length = int(sample_rate * 3)
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1
    result = []
    for channel in range(2):
        room = np.zeros(length, dtype=np.float32)
        for seconds in (.0297, .0371, .0411, .0437):
            delay = max(1, int(sample_rate * (seconds + channel * .00079)))
            output = np.zeros(length, dtype=np.float32)
            state = np.zeros(1)
            for start in range(0, length - delay, delay):
                size = min(delay, length - delay - start)
                filtered, state = lfilter([.75], [1, -.25], output[start:start + size], zi=state)
                output[start + delay:start + delay + size] = impulse[start:start + size] + .76 * filtered
            room += output * .25
        for seconds, offset in ((.005, .00041), (.0017, .00019)):
            delay = max(1, int(sample_rate * (seconds + channel * offset)))
            output = -.5 * room
            for start in range(delay, length, delay):
                size = min(delay, length - start)
                output[start:start + size] += room[start - delay:start - delay + size] + .5 * output[start - delay:start - delay + size]
            room = output
        room.flags.writeable = False
        result.append(room)
    return tuple(result)


def apply_reverb(audio, sample_rate, amount):
    if not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("Reverb must be between 0 and 1")
    if amount == 0:
        return audio
    kernels = room_impulses(sample_rate)
    output = audio * (1 - .3 * amount)
    for channel in range(2):
        source = .85 * audio[:, channel] + .15 * audio[:, 1 - channel]
        output[:, channel] += .7 * amount * fftconvolve(source, kernels[channel])[:len(audio)]
    return output.astype(np.float32, copy=False)
