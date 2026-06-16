"""D1 microwave_cycle -- appliance run start + authoritative end (A-solve).

Thin wrapper over the frozen DSP in detectors/detectors_lib.py (detect_hum_runs
+ detect_beeps, params loaded from results_hum_beep.json). The hum gives the run
start; the end-beep, when present near the hum offset, is the authoritative end
(the hum truncates at low SNR), else the hum offset is used.

Emits: cycle_start, cycle_end (with duration_s).
Latency: cycle_start ~8 s (driven by hum min_run_s=8), cycle_end ~8 s (the
end-beep association window). Both within the <=10 s online buffer.
"""
import os
import sys

from .base import Detector, event

# detectors_lib lives one level up (detectors/); it holds the frozen DSP + params.
_DET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DET_DIR not in sys.path:
    sys.path.insert(0, _DET_DIR)
import detectors_lib as dl  # noqa: E402


class MicrowaveCycleDetector(Detector):
    primitive = "D1"
    name = "microwave_cycle"
    role = "A-solve"
    gate = "recipe contains a microwave step"
    emits = ("cycle_start", "cycle_end", "duration")

    def __init__(self, beep_assoc_s=8.0, refractory_s=10.0):
        # beep_assoc_s caps how far past the hum offset we look for the end-beep.
        # The beep lands within ~1 s of the true offset, so 8 s catches the same
        # beep while keeping cycle_end within the <=10 s online buffer (was 20 s,
        # which violated the buffer). See tasks/AUDIO_RUNTIME_LIBRARY.md.
        # refractory_s merges hum fragments separated by < refractory_s into one
        # cycle: a real microwave cannot restart within ~10 s, so a brief hum dip
        # is the same run, not a new one. The gap is < the 10 s buffer, so the
        # merge decision is still causal/online.
        self.hb, _ = dl.load_frozen_params()
        self.beep_assoc_s = beep_assoc_s
        self.refractory_s = refractory_s

    def detect(self, x16, fs=16000):
        runs = dl.detect_hum_runs(x16, fs, self.hb)
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] < self.refractory_s:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append((r[0], r[1]))
        runs = merged
        beeps = dl.detect_beeps(x16, fs, self.hb)
        evs = []
        for start_s, hum_end in runs:
            # the microwave end is a beep CLUSTER; the last beep is the true offset
            # (matches eval/engine.fuse_done). Window [-5, +beep_assoc_s] stays within
            # the <=10 s buffer.
            cand = [b["t"] for b in beeps
                    if hum_end - 5.0 <= b["t"] <= hum_end + self.beep_assoc_s]
            end_s = max(cand) if cand else hum_end
            evs.append(event(start_s, self.primitive, "cycle_start"))
            evs.append(event(end_s, self.primitive, "cycle_end",
                             duration_s=round(end_s - start_s, 2)))
        return evs
