# Research / evidence docs — NOT runtime specs

These are the **provenance and evidence** behind the audio detectors: probe results,
EPIC-SOUNDS design rationale, the full annotated catalog (with refuted alternatives and
dated findings), and the stovetop survey. They record *how we decided what works*.

**The runtime system does NOT read these.** The clean, filtered, runtime-facing detector
vocabulary the compiler binds is **`../../tasks/AUDIO_RUNTIME_LIBRARY.md`** (only measured-reliable
detectors with ≤10 s decision windows). If a detail here disagrees with the runtime library,
the runtime library wins for anything the system executes.

| File | What it is |
|---|---|
| `AUDIO_RUNTIME_LIBRARY.md` (in `../../tasks/`) | **the runtime vocabulary** — start here for the system |
| `DETECTOR_CATALOG.md` | full annotated primitive catalog + dated findings (research log) |
| `DETECTOR_FEASIBILITY.md` | task inventory, per-detector probe evidence, cost ledger |
| `AUDIO_LIBRARY.md` | EPIC-SOUNDS-grounded design rationale for the audio primitives |
| `STOVETOP_AUDIO_SURVEY.md` | A4-vs-AL head-to-head, boundary/onset/fusion measurements |

Probe code + raw results: `detectors/probes/`.
