"""D2 appliance_motor -- blender / grinder on/off (A-solve, gated).

Ported from detectors/probes/appliance_water_eval.py (the `loud_runs` function
with the `APP` config). A wide-band sustained-level run well above the
per-recording quiet floor, with sharp on/off edges. Recipe-gating is required:
ungated it fires on other loud events.

Emits: motor_on, motor_off.
Latency: ~5-7 s.
"""
import numpy as np
from scipy.signal import stft

from .base import Detector, event

# frozen probe config (appliance_water_eval.py: APP) + the loud_runs defaults
APP_PARAMS = dict(band=(200.0, 8000.0), level_db=12.0, min_run_s=5.0,
                  roll_win_s=4.0, merge_s=3.0, block_s=0.5)


def _loud_runs(x, fs, band, level_db, min_run_s, roll_win_s=4.0, merge_s=3.0,
               block_s=0.5):
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
    w = max(1, int(round(roll_win_s / dtc)))
    half = w // 2
    roll = np.array([np.median(coarse[max(0, i - half):i + half + 1]) for i in range(nb)])
    floor = np.percentile(roll, 20)
    on = roll - floor > level_db
    runs, i = [], 0
    while i < nb:
        if on[i]:
            j = i
            while j + 1 < nb and on[j + 1]:
                j += 1
            runs.append([float(tc[i]) - dtc / 2, float(tc[j]) + dtc / 2])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_run_s]


class ApplianceMotorDetector(Detector):
    primitive = "D2"
    name = "appliance_motor"
    role = "A-solve (gated)"
    gate = "recipe contains a blender/grinder step; expect one motor event near it"
    emits = ("motor_on", "motor_off")

    def __init__(self, **overrides):
        self.p = {**APP_PARAMS, **overrides}

    def detect(self, x16, fs=16000):
        runs = _loud_runs(x16, fs, **self.p)
        evs = []
        for start_s, end_s in runs:
            evs.append(event(start_s, self.primitive, "motor_on"))
            evs.append(event(end_s, self.primitive, "motor_off",
                             duration_s=round(end_s - start_s, 2)))
        return evs
