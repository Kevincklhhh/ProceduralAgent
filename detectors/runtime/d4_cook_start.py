"""D4 cook_start -- onset of frying (B-trigger).

Ported from detectors/probes/onset_eval.py (`sizzle_onsets`). A causal rising
edge of the 1.5-7 kHz band energy against a trailing 30 s median, fired when it
jumps +8 dB and holds >=5 s. Frying ramps in with no sharp onset, so this is a
trigger (fire one VLM call to confirm cook start), never an A-solve.

Emits: cook_start_candidate.
Latency: ~5-8 s.
"""
import numpy as np
from scipy.signal import stft

from .base import Detector, event

# frozen probe params (onset_eval.py)
ONSET_PARAMS = dict(band=(1500.0, 7000.0), block_s=0.5, trail_s=30.0,
                    min_hist_s=8.0, level_db=8.0, hold_s=5.0)


def _sizzle_onsets(x, fs, band, block_s, trail_s, min_hist_s, level_db, hold_s):
    f, t, Z = stft(x, fs=fs, nperseg=1024, noverlap=1024 - 256, window='hann',
                   padded=False, boundary=None)
    P = np.abs(Z) ** 2
    bm = (f >= band[0]) & (f <= band[1])
    bdb = 10 * np.log10(P[bm].sum(axis=0) + 1e-12)
    dt = 256 / fs
    blk = max(1, int(round(block_s / dt)))
    nb = len(bdb) // blk
    if nb < 3:
        return []
    coarse = np.median(bdb[:nb * blk].reshape(nb, blk), axis=1)
    tc = (np.arange(nb) + 0.5) * blk * dt
    dtc = blk * dt
    trail = int(trail_s / dtc)
    hist = int(min_hist_s / dtc)
    hold = int(hold_s / dtc)
    onsets, i = [], hist
    while i < nb - hold:
        base = np.median(coarse[max(0, i - trail):i])
        if coarse[i] - base > level_db and np.mean(coarse[i:i + hold]) - base > level_db - 2:
            onsets.append(float(tc[i]))
            i += hold + trail
        else:
            i += 1
    return onsets


class CookStartDetector(Detector):
    primitive = "D4"
    name = "cook_start"
    role = "B-trigger"
    gate = "recipe contains a fry stage"
    emits = ("cook_start_candidate",)

    def __init__(self, **overrides):
        self.p = {**ONSET_PARAMS, **overrides}

    def detect(self, x16, fs=16000):
        onsets = _sizzle_onsets(x16, fs, **self.p)
        return [event(o, self.primitive, "cook_start_candidate") for o in onsets]
