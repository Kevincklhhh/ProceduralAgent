"""Shared PANNs CNN14 (AudioSet) front-end for the learned detectors D3 and D5.

One model, loaded once and cached, exposing a pooled-probability signal over a
chosen set of AudioSet class indices via 2 s / 1 s sliding-window clip-level
tagging. This replaces the copy-pasted loader that lived in probe_panns_sizzle.py
and water_cnn_eval.py.

Requires torch + panns_inference + the checkpoint at ~/panns_data/. torch is
imported lazily so importing the runtime package stays cheap (D1/D2/D4/D6 are
DSP/logic only and need no GPU).
"""
import os

import numpy as np
from scipy.signal import resample_poly

CKPT = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'

# AudioSet class index sets (verbatim from the probes)
COOK_IDX = [367, 490, 456, 296]    # Frying, Sizzle, Boiling, Steam
WATER_IDX = [288, 370, 371, 444]   # Water, Water tap/faucet, Sink, Liquid

_MODEL = None
_DEV = None


def load_model():
    """Load CNN14 once on a single device; cached for reuse across detectors."""
    global _MODEL, _DEV
    if _MODEL is not None:
        return _MODEL, _DEV
    import torch
    from panns_inference.models import Cnn14
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
    model.load_state_dict(torch.load(CKPT, map_location='cpu')['model'])
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(dev).eval()
    _MODEL, _DEV = model, dev
    return model, dev


def _pooled_signal(x48, class_idx, win=2.0, hop=1.0, batch=64):
    """(t, prob): prob = max over class_idx of the clipwise output on a `win`s
    window stepped `hop`s; t = window centers. `x48` is the normalised 48 kHz
    stream (resampled 2/3 -> 32 kHz for the model)."""
    import torch
    model, dev = load_model()
    x32 = resample_poly(x48, 2, 3).astype(np.float32)
    W, H = int(win * 32000), int(hop * 32000)
    starts = list(range(0, max(1, len(x32) - W), H))
    probs = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            segs = np.stack([x32[s:s + W] for s in starts[i:i + batch]])
            out = model(torch.from_numpy(segs).to(dev), None)['clipwise_output'].cpu().numpy()
            probs.append(out[:, class_idx].max(axis=1))
    sig = np.concatenate(probs) if probs else np.zeros(0)
    t = np.array(starts[:len(sig)]) / 32000.0 + win / 2
    return t, sig


def cook_signal(x48, **kw):
    """Cook-active probability (Frying/Sizzle/Boiling/Steam). Used by D3."""
    return _pooled_signal(x48, COOK_IDX, **kw)


def water_signal(x48, **kw):
    """Water-flow probability (Water/Tap/Sink/Liquid). Used by D5."""
    return _pooled_signal(x48, WATER_IDX, **kw)
