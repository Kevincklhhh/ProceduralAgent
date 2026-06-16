"""Runtime detector library (D1-D6) for the procedure monitor.

One module per detector in the catalog tasks/AUDIO_RUNTIME_LIBRARY.md. Each
detector emits role-agnostic events shaped to the procedure_monitor
output_schema.sensor_events entry; the monitor decides what to do with them
(A-solve = settle the step, B-trigger = fire one VLM call, C-none = periodic
VLM). The VLM itself is the single expensive sensor (vlm.py, stubbed).

Provenance: the DSP/learned code is ported from the validated frozen probes
under detectors/probes/ and detectors/detectors_lib.py. No re-tuning here.
"""
from .base import Detector, event
from .d1_microwave_cycle import MicrowaveCycleDetector
from .d2_appliance_motor import ApplianceMotorDetector
from .d3_cook_end import CookEndDetector
from .d4_cook_start import CookStartDetector
from .d5_water_flow import WaterFlowDetector
from .d6_timer import TimerChecker
from .vlm import VLMClient

# D# -> detector class, as bound in tasks/AUDIO_RUNTIME_LIBRARY.md
REGISTRY = {
    "D1": MicrowaveCycleDetector,
    "D2": ApplianceMotorDetector,
    "D3": CookEndDetector,
    "D4": CookStartDetector,
    "D5": WaterFlowDetector,
    "D6": TimerChecker,
}

__all__ = [
    "Detector", "event", "REGISTRY", "VLMClient",
    "MicrowaveCycleDetector", "ApplianceMotorDetector", "CookEndDetector",
    "CookStartDetector", "WaterFlowDetector", "TimerChecker",
]
