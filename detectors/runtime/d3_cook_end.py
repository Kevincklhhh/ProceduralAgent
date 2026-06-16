"""D3 cook_end -- end of frying / saute / boil (A-solve, END only).

Ported from detectors/probes/probe_panns_sizzle.py. Uses the shared PANNs CNN14
cook-active probability (panns.cook_signal). NOT a start and NOT a coverage signal.

cook_end is BOUNDED for the online buffer: rather than "the last above-threshold
frame of the whole clip" (which can't be known until the stream ends), we emit a
cook_end at the end of any cooking segment that is followed by `hangover_s` of
below-threshold cook probability (or by end-of-stream). The lookahead is therefore
hangover_s (<=10 s), matching the measured ~8 s cook-end offset.

Emits: cook_end (one per confirmed cooking-segment end).
Latency: ~hangover_s.

`threshold` default 0.1 matches the catalog; the probe also explored 0.2 for the
end-of-cook decision. Both exposed so they can be tuned without editing code.
"""
from .base import Detector, event


class CookEndDetector(Detector):
    primitive = "D3"
    name = "cook_end"
    role = "A-solve (END only)"
    gate = "recipe contains a fry/saute/boil stage"
    emits = ("cook_end",)

    def __init__(self, threshold=0.1, hangover_s=8.0):
        self.threshold = threshold
        self.hangover_s = hangover_s

    def detect(self, x48):
        from . import panns
        t, sig = panns.cook_signal(x48)
        n = len(sig)
        if n == 0:
            return []
        on = sig > self.threshold
        dt = float(t[1] - t[0]) if n > 1 else 1.0
        hang = max(1, int(round(self.hangover_s / dt)))
        evs, i = [], 0
        while i < n:
            if on[i]:
                j = i
                while j + 1 < n and on[j + 1]:
                    j += 1
                # confirm this segment's end once `hang` off-frames follow, or the
                # stream ends while still cooking (trailing segment).
                off_after, k = 0, j + 1
                while k < n and not on[k] and off_after < hang:
                    off_after += 1
                    k += 1
                if off_after >= hang or k >= n:
                    evs.append(event(t[j], self.primitive, "cook_end",
                                     confidence=float(sig[j])))
                i = j + 1
            else:
                i += 1
        return evs
