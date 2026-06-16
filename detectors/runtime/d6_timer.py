"""D6 timer -- duration & precondition logic (A-solve, pure logic).

Generalised from the inline checks in eval/engine.py (over/under-time on the
microwave runs, and the missing-mix-before-heat precondition). No audio: it
reasons over the procedure graph plus completion events emitted by D1-D5.

Emits: overtime, undertime, precondition_violation.
Latency: none (logic).
"""
from .base import Detector, event


class TimerChecker(Detector):
    primitive = "D6"
    name = "timer"
    role = "A-solve (logic)"
    gate = "timed steps / precondition edges"
    emits = ("overtime", "undertime", "precondition_violation")

    def __init__(self, tol=0.33):
        # duration bound = stated value +/- tol (engine used +/-33%)
        self.tol = tol

    def check_duration(self, step_id, start_s, end_s, expected_s):
        """Return an overtime/undertime event, or None if within bounds."""
        dur = end_s - start_s
        lo, hi = expected_s * (1 - self.tol), expected_s * (1 + self.tol)
        if dur < lo:
            return event(end_s, self.primitive, "undertime", confidence=1.0,
                         step_id=step_id, duration_s=round(dur, 1),
                         expected_s=expected_s)
        if dur > hi:
            return event(start_s + hi, self.primitive, "overtime", confidence=1.0,
                         step_id=step_id, duration_s=round(dur, 1),
                         expected_s=expected_s)
        return None

    def check_precondition(self, step_id, start_s, requires, completed_steps):
        """Return a precondition_violation if any required step isn't complete."""
        missing = [r for r in requires if r not in completed_steps]
        if missing:
            return event(start_s, self.primitive, "precondition_violation",
                         confidence=1.0, step_id=step_id, missing=missing)
        return None
