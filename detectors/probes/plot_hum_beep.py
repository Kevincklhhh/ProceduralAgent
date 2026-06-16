"""Overview figure for the hum_beep probe: F1 trace, detected runs, GT windows, beeps."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes')
import hum_beep as hb

gt = json.load(open(hb.GT_PATH))
res = json.load(open(hb.OUT_PATH))
recs = ['8_16', '8_26', '8_3', '8_25', '8_31', '8_50']

fig, axes = plt.subplots(len(recs), 1, figsize=(16, 2.6 * len(recs)))
for ax, rec in zip(axes, recs):
    fs, x = hb.load_audio(rec)
    t, f1, f2, f3 = hb.hum_features(x, fs, hb.PARAMS)
    d = (res['tuned_recording_results'] if rec == '8_16' else res['frozen_eval_results'])[rec]
    ax.plot(t, f1, lw=0.6, color='steelblue', label='F1 level (dB>base, smoothed)')
    ax.plot(t, f2, lw=0.6, color='seagreen', alpha=0.7, label='F2 120Hz line (dB)')
    for s in gt[rec]['steps']:
        if s['step_id'] in (89, 83) and s['start_time'] >= 0:
            ax.axvspan(s['start_time'], s['end_time'], color='orange', alpha=0.18,
                       label='GT microwave step' if s['step_id'] == 89 else None)
    for i, (a, b, _) in enumerate(d['hum_runs']):
        ax.axvspan(a, b, ymin=0, ymax=0.12, color='red', alpha=0.8,
                   label='detected hum run' if i == 0 else None)
    bt = [bp['t'] for bp in d['beeps']]
    ax.plot(bt, [26] * len(bt), 'v', color='purple', ms=5, label='beep')
    ax.set_ylim(-4, 30)
    ax.set_ylabel(rec)
    if rec == recs[0]:
        ax.legend(loc='upper right', fontsize=7, ncol=5)
axes[-1].set_xlabel('time (s)')
fig.suptitle('hum_beep probe: microwave hum runs + beeps vs GT (orange = GT steps 89/83)')
plt.tight_layout()
plt.savefig('/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes/hum_beep_overview.png', dpi=80)
print('saved hum_beep_overview.png')
