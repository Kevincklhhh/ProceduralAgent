"""Common event schema and detector base for the runtime library.

Roles (A-solve / B-trigger / C-none) are deliberately NOT encoded in a
detector. A detector only reports what it observed; the monitor decides what to
do with the event. The `role` attribute below is informational metadata only,
matching the catalog in tasks/AUDIO_RUNTIME_LIBRARY.md.
"""


def event(t_s, primitive, name, confidence=1.0, **extra):
    """Build one sensor event matching output_schema.sensor_events.

    Extra fields (e.g. duration_s, end_s, step_id) are merged in for downstream
    state updates. DSP detectors have no calibrated probability, so they pass
    confidence=1.0; the learned detectors (D3/D5) pass the model probability.
    """
    e = {
        "t_s": round(float(t_s), 2),
        "primitive": primitive,
        "event": name,
        "confidence": round(float(confidence), 3),
    }
    e.update(extra)
    return e


class Detector:
    """Base class. Subclasses set the metadata fields and implement detect()."""

    primitive = None      # "D1".."D6"
    name = None           # catalog name, e.g. "microwave_cycle"
    role = None           # informational: "A-solve" | "B-trigger" | ...
    gate = None           # one-line gate condition from the catalog
    emits = ()            # event names this detector can emit

    def detect(self, *args, **kwargs):
        raise NotImplementedError
