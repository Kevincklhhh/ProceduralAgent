"""Shared audio front-end. One continuous 16 kHz (DSP) / 48 kHz (PANNs) stream.

Verbatim from detectors_lib.load_audio_16k / load_audio_48k so the runtime reads
exactly the same WAVs the probes were validated on.
"""
import numpy as np
from scipy.io import wavfile

AUDIO_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio'


def load_audio_16k(rec):
    fs, x = wavfile.read(f'{AUDIO_DIR}/{rec}_16k.wav')
    return fs, x.astype(np.float64) / 32768.0


def load_audio_48k(rec):
    fs, x = wavfile.read(f'{AUDIO_DIR}/{rec}_48k.wav')
    return fs, x.astype(np.float64) / 32768.0
