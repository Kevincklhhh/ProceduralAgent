"""The expensive sensor: a stubbed VLM client.

Detectors never call this. The monitor does: a B-trigger step fires exactly one
call at the flagged event; a C-none step calls it on a periodic schedule. This
keeps the VLM asleep except where audio cannot decide -- the energy/latency win.

Replace `query` with a real model (e.g. Qwen on saltyfish). The stub records
call count so cost.vlm_calls can be tallied in the meantime.
"""


class VLMClient:
    def __init__(self, model="stub"):
        self.model = model
        self.calls = 0

    def query(self, t_s, frames=None, prompt=""):
        """Return a step verdict for the monitor. STUB: no model wired yet."""
        self.calls += 1
        return {
            "t_s": round(float(t_s), 2),
            "model": self.model,
            "answer": None,
            "note": "STUB: wire a real VLM here",
        }
