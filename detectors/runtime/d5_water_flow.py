"""D5 water_flow -- tap on/off (B-trigger / gated A-solve).

Ported from detectors/probes/water_cnn_eval.py. Uses the shared PANNs CNN14
water probability (panns.water_signal) -- the same model as D3, so it is free
wherever D3 already runs. Recipes have several water uses, so bind to one
specific rinse/wash/fill step rather than every tap event.

Emits: water_on, water_off.
Latency: ~2 s.
"""
from .base import Detector, event


def _intervals(t, on, gap=3.0, mindur=3.0):
    """Verbatim from water_cnn_eval.intervals: merge on-frames into [start,end]."""
    import numpy as np
    idx = np.where(on)[0]
    if not len(idx):
        return []
    runs, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            runs.append([t[s], t[a]])
            s = b
    runs.append([t[s], t[idx[-1]]])
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [[a, b] for a, b in merged if b - a >= mindur]


class WaterFlowDetector(Detector):
    primitive = "D5"
    name = "water_flow"
    role = "B-trigger / gated A-solve"
    gate = "bind to the specific rinse/wash/fill step"
    emits = ("water_on", "water_off")

    def __init__(self, threshold=0.1, gap_s=3.0, min_dur_s=3.0):
        self.threshold = threshold
        self.gap_s = gap_s
        self.min_dur_s = min_dur_s

    def detect(self, x48):
        from . import panns
        t, sig = panns.water_signal(x48)
        ivs = _intervals(t, sig > self.threshold, self.gap_s, self.min_dur_s)
        evs = []
        for start_s, end_s in ivs:
            evs.append(event(start_s, self.primitive, "water_on"))
            evs.append(event(end_s, self.primitive, "water_off",
                             duration_s=round(end_s - start_s, 2)))
        return evs
