#!/usr/bin/env python3
"""Focused CV probe for Spiced Hot Chocolate criteria.

Scope (v1): skip fill_milk and mix. Probe only:
  1. generic add-event segmentation in the unordered add block,
  2. chocolate-piece count as a measurement-error cue,
  3. microwave display OCR as a visual cross-check for timing.

The broad sweep is intentionally cheap/OpenCV-first over all 16 activity-8 recordings.
Optional local model probes (GroundingDINO / OWLv2 / YOLO-World) run on a small number of
frames to inspect whether higher-capacity CV can help; they are not required for the
main add-event score.
"""
import argparse
import json
import os
import re
import time

import cv2
import numpy as np


BASE = "/home/kailaic/NeuroTrace/ProceduralAgent"
VID360 = os.path.join(BASE, "data", "videos_360p")
VID480 = os.path.join(BASE, "data", "videos_480p")
ANN_PATH = os.path.join(
    BASE, "data", "cc4d", "annotations", "annotation_json", "complete_step_annotations.json"
)
ERROR_ANN_PATH = os.path.join(
    BASE, "data", "cc4d", "annotations", "annotation_json", "error_annotations.json"
)
PROBE_DIR = os.path.join(BASE, "detectors", "probes")
OUT_PATH = os.path.join(PROBE_DIR, "results_cv_shc_criteria.json")

SHC_RECS = [
    "8_11", "8_15", "8_16", "8_19", "8_20", "8_25", "8_26", "8_3",
    "8_30", "8_31", "8_33", "8_35", "8_40", "8_44", "8_45", "8_50",
]
DIAGNOSTIC_480_RECS = {"8_16", "8_3", "8_25", "8_26", "8_31", "8_50"}
CHOCOLATE_TARGET_RECS = {"8_26", "8_30", "8_33", "8_35", "8_44", "8_45"}

ADD_STEPS = (90, 84, 87)  # chocolate, cinnamon, sugar
CHOCOLATE_STEP = 90
MICROWAVE_STEPS = (89, 83)

SAMPLE_FPS = 3.0
PAD_S = 8.0
TOL_S = 15.0

# Same fixed lower-center mug ROI as probe_rgb_adds.py. This is a stand-in for a
# future one-shot grounded mug ROI.
MUG_ROI = (0.30, 0.72, 0.34, 0.66)

# Existing baseline from probe_rgb_adds.py.
BASELINE_CHI2_MIN = 2.5
BASELINE_MINSEP_S = 15.0

# Focused v1 setting: same state-change threshold, wider episode separation.
# Tuned on cached features to keep recall >=0.90 while improving precision.
FOCUSED_CHI2_MIN = 2.5
FOCUSED_MINSEP_S = 30.0


ANN = json.load(open(ANN_PATH))
try:
    _ERROR_ITEMS = json.load(open(ERROR_ANN_PATH))
except FileNotFoundError:
    _ERROR_ITEMS = []
ERROR_STEPS = {
    item["recording_id"]: {step["step_id"]: step for step in item.get("step_annotations", [])}
    for item in _ERROR_ITEMS
}


def video_path(rec, prefer_480=False):
    if prefer_480 and rec in DIAGNOSTIC_480_RECS:
        path_480 = os.path.join(VID480, f"{rec}.mp4")
        if os.path.exists(path_480):
            return path_480, "480p"
    return os.path.join(VID360, f"{rec}.mp4"), "360p"


def rounded(x, n=3):
    if x is None:
        return None
    return round(float(x), n)


def crop_frac(frame, frac):
    r0, r1, c0, c1 = frac
    h, w = frame.shape[:2]
    return frame[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]


def hsv_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def step_map(rec):
    return {s["step_id"]: s for s in ANN[rec]["steps"]}


def present_step_window(rec, step_id):
    step = step_map(rec).get(step_id)
    if not step or step["start_time"] < 0:
        return None
    return float(step["start_time"]), float(step["end_time"])


def add_gt(rec):
    steps = step_map(rec)
    gt = []
    for sid in ADD_STEPS:
        step = steps.get(sid)
        if step and step["start_time"] >= 0:
            gt.append((sid, float(step["start_time"]), float(step["end_time"])))
    return sorted(gt, key=lambda x: x[1])


def add_window(rec):
    gt = add_gt(rec)
    if not gt:
        return None
    return max(0.0, gt[0][1] - PAD_S), max(e for _, _, e in gt) + PAD_S


def sample_times(start, end, fps=None, step_s=None, max_frames=None):
    if end < start:
        return []
    if step_s is None:
        step_s = 1.0 / (fps or SAMPLE_FPS)
    times = []
    t = start
    while t <= end + 1e-6:
        times.append(float(t))
        t += step_s
    if max_frames and len(times) > max_frames:
        idx = np.linspace(0, len(times) - 1, max_frames).round().astype(int)
        times = [times[i] for i in sorted(set(idx))]
    return times


def read_frame(video_path, t_s):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def read_frames(video_path, times):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            out.append((float(t), frame))
    cap.release()
    return out


def load_add_features(rec):
    """Return t and HSV histograms for the add block, reusing old cache if present."""
    win = add_window(rec)
    if win is None:
        return np.array([]), np.zeros((0, 256))

    old_cache = os.path.join(PROBE_DIR, f"rgb_adds_{rec}.npz")
    if os.path.exists(old_cache):
        data = np.load(old_cache, allow_pickle=True)
        return data["t"], data["hists"]

    video, _ = video_path(rec)
    times = sample_times(win[0], win[1], fps=SAMPLE_FPS)
    ts, hists = [], []
    for t, frame in read_frames(video, times):
        ts.append(t)
        hists.append(hsv_hist(crop_frac(frame, MUG_ROI)))
    return np.array(ts), np.array(hists)


def chi2_series(hists):
    d = int(3 * SAMPLE_FPS)
    gap = int(1 * SAMPLE_FPS)
    chi = np.zeros(len(hists))
    for i in range(len(hists)):
        if i - gap <= 0 or i + gap + d > len(hists):
            continue
        pre = hists[max(0, i - gap - d):i - gap].mean(0)
        post = hists[i + gap:i + gap + d].mean(0)
        chi[i] = 0.5 * np.sum((pre - post) ** 2 / (pre + post + 1e-9))
    return chi


def find_peaks(t, score, threshold, minsep_s):
    order = np.argsort(score)[::-1]
    peaks = []
    for i in order:
        if score[i] < threshold:
            break
        if all(abs(t[i] - t[j]) > minsep_s for j in peaks):
            peaks.append(i)
    return sorted(peaks)


def match_peaks(peak_ts, gt, tol=TOL_S):
    used = set()
    pairs = []
    for pt in peak_ts:
        best = None
        for k, (sid, s, e) in enumerate(gt):
            if k in used:
                continue
            if s - tol <= pt <= e + tol:
                best = k
                break
        if best is not None:
            used.add(best)
            sid, s, e = gt[best]
            pairs.append({
                "peak_t": rounded(pt, 1),
                "gt_step": sid,
                "gt_window": [rounded(s, 1), rounded(e, 1)],
            })
    return len(used), pairs


def score_add_events(rec):
    gt = add_gt(rec)
    if not gt:
        return {
            "expected_adds": 0,
            "gt_present_steps": [],
            "baseline": {},
            "focused": {},
            "missing_add_signal": {"gt_missing_adds": 3, "pred_missing_add": True},
        }

    t, hists = load_add_features(rec)
    chi = chi2_series(hists)

    def run_variant(threshold, minsep_s):
        peaks = find_peaks(t, chi, threshold, minsep_s)
        peak_ts = [float(t[i]) for i in peaks]
        tp, pairs = match_peaks(peak_ts, gt)
        return {
            "threshold": threshold,
            "minsep_s": minsep_s,
            "n_peaks": len(peaks),
            "matched_adds_TP": tp,
            "spurious_peaks": len(peaks) - tp,
            "recall": rounded(tp / len(gt)) if gt else None,
            "precision": rounded(tp / len(peaks)) if peaks else None,
            "peak_times": [rounded(x, 1) for x in peak_ts],
            "matches": pairs,
        }

    baseline = run_variant(BASELINE_CHI2_MIN, BASELINE_MINSEP_S)
    focused = run_variant(FOCUSED_CHI2_MIN, FOCUSED_MINSEP_S)
    gt_missing = 3 - len(gt)
    pred_missing = focused["n_peaks"] < 3
    return {
        "window": [rounded(x, 1) for x in add_window(rec)],
        "model_used": "opencv_hsv_mug_roi_state_change",
        "confidence": "medium" if focused.get("precision", 0) and focused.get("precision", 0) >= 0.60 else "low",
        "failure_reason": None if len(t) else "no_sampled_frames",
        "video_resolution": "360p",
        "sampled_frames": int(len(t)),
        "sampled_frame_times": [rounded(x, 1) for x in t.tolist()],
        "expected_adds": len(gt),
        "gt_present_steps": [[sid, rounded(s, 1), rounded(e, 1)] for sid, s, e in gt],
        "baseline": baseline,
        "focused": focused,
        "missing_add_signal": {
            "gt_missing_adds": gt_missing,
            "pred_missing_add": bool(pred_missing),
            "detected_add_episodes": focused["n_peaks"],
            "verdict": "TP" if pred_missing and gt_missing else
                       "FP" if pred_missing and not gt_missing else
                       "FN" if (not pred_missing and gt_missing) else "TN",
            "note": "episode-count signal only; does not identify which ingredient is missing",
        },
    }


def expected_chocolate_count(rec):
    step = step_map(rec).get(CHOCOLATE_STEP)
    if not step or step["start_time"] < 0:
        return None

    err_step = ERROR_STEPS.get(rec, {}).get(CHOCOLATE_STEP, {})
    parts = [
        err_step.get("modified_description", ""),
        " ".join(e.get("description", "") for e in err_step.get("errors", []) or []),
        err_step.get("description", ""),
        step.get("modified_description", ""),
        " ".join(e.get("description", "") for e in step.get("errors", []) or []),
        step.get("description", ""),
    ]
    text = " ".join(x for x in parts if x).lower()

    patterns = [
        r"add(?:ed)?[- ]add\s+(\d+)\s+pieces?\s+of\s+chocolate",
        r"add(?:ed)?\s+(\d+)\s+pieces?\s+of\s+chocolate",
        r"(\d+)\s+pieces?\s+of\s+chocolate",
        r"(\d+)\s+piece\s+of\s+chocolate",
        r"only\s+one\s+piece",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return 1 if pat.startswith("only") else int(m.group(1))
    return 2


def dark_blob_count(frame):
    """Crude chocolate-piece proxy in the fixed mug ROI."""
    roi = crop_frac(frame, MUG_ROI)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Dark, moderately saturated brown/black candidates. This intentionally avoids
    # high-level semantics; it is a cheap measurement cue, not ingredient identity.
    mask = ((v < 105) & (s > 35)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    area_total = 0
    for i in range(1, n):
        x, y, w, hgt, area = stats[i]
        if 12 <= area <= 1800 and w >= 3 and hgt >= 3:
            comps.append((int(x), int(y), int(w), int(hgt), int(area)))
            area_total += int(area)
    comps = sorted(comps, key=lambda item: item[-1], reverse=True)
    return len(comps), area_total, comps[:3]


def score_chocolate_count(rec):
    win = present_step_window(rec, CHOCOLATE_STEP)
    expected = expected_chocolate_count(rec)
    if win is None:
        return {
            "present": False,
            "expected_count": expected,
            "status": "skipped",
        }

    video, resolution = video_path(rec)
    times = sample_times(max(0.0, win[0] - PAD_S), win[1] + PAD_S, fps=SAMPLE_FPS, max_frames=64)
    per_frame = []
    for t, frame in read_frames(video, times):
        cnt, area, comps = dark_blob_count(frame)
        per_frame.append({
            "t": rounded(t, 1),
            "dark_components": cnt,
            "dark_area": area,
            "top_components": comps,
        })

    counts = [x["dark_components"] for x in per_frame]
    if counts:
        # A robust "how many discrete dark pieces were visible at peak" proxy.
        pred = int(np.percentile(counts, 90))
        pred = min(pred, 8)
    else:
        pred = None

    if pred is None or expected is None:
        verdict = "unknown"
    elif pred == expected:
        verdict = "exact"
    elif (pred < 2 and expected < 2) or (pred > 2 and expected > 2):
        verdict = "same_side_of_nominal"
    else:
        verdict = "wrong_side_of_nominal"

    top = sorted(per_frame, key=lambda x: (x["dark_components"], x["dark_area"]), reverse=True)[:5]
    return {
        "present": True,
        "status": "diagnostic_proxy",
        "model_used": "opencv_dark_blob_mug_roi_proxy",
        "failure_reason": None if per_frame else "no_readable_frames",
        "confidence": "low",
        "window": [rounded(win[0], 1), rounded(win[1], 1)],
        "video_resolution": resolution,
        "sampled_frames": len(per_frame),
        "sampled_frame_times": [rounded(t, 1) for t in times],
        "expected_count": expected,
        "predicted_dark_blob_count": pred,
        "verdict": verdict,
        "top_frames": top,
        "note": "dark-blob count is a low-confidence non-semantic proxy; use object-model diagnostics before treating it as an acceptance metric",
    }


class ObjectProbe:
    def __init__(self, mode):
        self.mode = mode
        self.ready = False
        self.error = None
        self.labels = ["mug", "hand", "chocolate pieces", "spoon", "microwave display"]
        self.model = None
        self.processor = None

        if mode == "none":
            return
        try:
            if mode in ("groundingdino", "owlv2"):
                import torch
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                model_id = {
                    "groundingdino": "IDEA-Research/grounding-dino-tiny",
                    "owlv2": "google/owlv2-base-patch16-ensemble",
                }[mode]
                self.torch = torch
                self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_id, local_files_only=True
                )
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                self.model_id = model_id
                self.ready = True
            elif mode == "yoloworld":
                from ultralytics import YOLOWorld

                weights = "/home/kailaic/.cache/bench_tmp/yolov8s-worldv2.pt"
                self.model = YOLOWorld(weights)
                self.model.set_classes(self.labels)
                self.model_id = weights
                self.ready = True
            else:
                self.error = f"unknown mode {mode}"
        except Exception as exc:  # noqa: BLE001 - diagnostics should not kill the probe.
            self.error = f"{type(exc).__name__}: {exc}"

    def detect(self, frame):
        if not self.ready:
            return []
        if self.mode == "yoloworld":
            res = self.model.predict(frame, verbose=False, conf=0.05, imgsz=640)[0]
            out = []
            for b in res.boxes:
                cls = int(b.cls[0])
                out.append({
                    "label": self.model.names[cls],
                    "score": rounded(float(b.conf[0])),
                    "box": [rounded(float(x), 1) for x in b.xyxy[0]],
                })
            return out

        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if self.mode == "groundingdino":
            text = [". ".join(self.labels) + "."]
            inputs = self.processor(images=image, text=text, return_tensors="pt")
        else:
            inputs = self.processor(images=image, text=[self.labels], return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        kwargs = {"threshold": 0.20, "target_sizes": [image.size[::-1]]}
        if self.mode == "groundingdino":
            kwargs["input_ids"] = inputs.get("input_ids")
            kwargs["text_threshold"] = 0.20
            kwargs["text_labels"] = [self.labels]
        else:
            kwargs["text_labels"] = [self.labels]
        res = self.processor.post_process_grounded_object_detection(outputs, **kwargs)[0]
        label_key = "text_labels" if "text_labels" in res else "labels"
        out = []
        for score, label, box in zip(res["scores"], res[label_key], res["boxes"]):
            out.append({
                "label": str(label),
                "score": rounded(float(score)),
                "box": [rounded(float(x), 1) for x in box],
            })
        return out


def box_center_in_frac(box, frac, frame_shape):
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0 / max(w, 1)
    cy = (y0 + y1) / 2.0 / max(h, 1)
    r0, r1, c0, c1 = frac
    return r0 <= cy <= r1 and c0 <= cx <= c1


def object_model_diagnostics(rec, probe, max_frames):
    if not probe.ready:
        return {"enabled": probe.mode, "ready": False, "error": probe.error}
    win = present_step_window(rec, CHOCOLATE_STEP)
    if win is None:
        return {"enabled": probe.mode, "ready": True, "note": "chocolate step skipped"}
    video, resolution = video_path(rec, prefer_480=True)
    times = sample_times(max(0.0, win[0] - PAD_S), win[1] + PAD_S,
                         fps=1.0, max_frames=max_frames)
    frames = []
    for t, frame in read_frames(video, times):
        dets = probe.detect(frame)
        choc = [
            d for d in dets
            if "chocolate" in d["label"].lower()
            and box_center_in_frac(d["box"], MUG_ROI, frame.shape)
        ]
        frames.append({
            "t": rounded(t, 1),
            "n_detections": len(dets),
            "n_chocolate_in_mug_roi": len(choc),
            "top": sorted(dets, key=lambda x: x["score"], reverse=True)[:8],
        })
    return {
        "enabled": probe.mode,
        "ready": True,
        "model": probe.model_id,
        "video_resolution": resolution,
        "sampled_frame_times": [rounded(t, 1) for t in times],
        "frames": frames,
    }


class OcrProbe:
    def __init__(self, enabled):
        self.enabled = enabled
        self.ready = False
        self.error = None
        self.reader = None
        if not enabled:
            return
        try:
            import easyocr

            try:
                self.reader = easyocr.Reader(["en"], gpu=True, verbose=False, download_enabled=False)
            except TypeError:
                self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            self.ready = True
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"

    def read_digits(self, frame):
        if not self.ready:
            return []
        # Full-frame OCR is noisy but robust to unknown microwave/display location.
        results = self.reader.readtext(frame, detail=1, paragraph=False, allowlist="0123456789:")
        out = []
        for box, text, conf in results:
            clean = re.sub(r"[^0-9:]", "", text)
            if clean:
                out.append({"text": clean, "confidence": rounded(conf),
                            "box": [[rounded(x, 1) for x in pt] for pt in box]})
        return out


def score_microwave_ocr(rec, ocr_probe, step_s, max_frames):
    out = {}
    if not ocr_probe.enabled:
        return {"enabled": False, "model_used": "easyocr"}
    if not ocr_probe.ready:
        return {"enabled": True, "ready": False, "model_used": "easyocr", "error": ocr_probe.error}

    video, resolution = video_path(rec)
    for sid in MICROWAVE_STEPS:
        win = present_step_window(rec, sid)
        key = str(sid)
        if win is None:
            out[key] = {"present": False, "status": "skipped"}
            continue
        times = sample_times(max(0.0, win[0] - 5.0), win[1] + 5.0,
                             step_s=step_s, max_frames=max_frames)
        hits = []
        for t, frame in read_frames(video, times):
            tokens = ocr_probe.read_digits(frame)
            usable = [x for x in tokens if re.search(r"\d", x["text"])]
            if usable:
                hits.append({"t": rounded(t, 1), "tokens": usable[:5]})
        out[key] = {
            "present": True,
            "model_used": "easyocr",
            "window": [rounded(win[0], 1), rounded(win[1], 1)],
            "video_resolution": resolution,
            "sampled_frames": len(times),
            "sampled_frame_times": [rounded(t, 1) for t in times],
            "usable_reads": len(hits),
            "status": "usable" if hits else "not_visible",
            "confidence": "medium" if hits else "low",
            "failure_reason": None if hits else "not_visible_or_unreadable_display",
            "hits": hits[:10],
        }
    return {"enabled": True, "ready": True, "model_used": "easyocr", "steps": out}


def aggregate(per_recording):
    base_exp = base_tp = base_peaks = 0
    foc_exp = foc_tp = foc_peaks = 0
    miss_tp = miss_fp = miss_fn = miss_tn = 0
    choc_exact = choc_known = choc_side = 0
    chocolate_targets = {}
    ocr_present = ocr_usable = 0

    for rec_res in per_recording.values():
        add = rec_res["add_events"]
        exp = add["expected_adds"]
        base = add["baseline"]
        foc = add["focused"]
        base_exp += exp
        foc_exp += exp
        base_tp += base.get("matched_adds_TP", 0)
        foc_tp += foc.get("matched_adds_TP", 0)
        base_peaks += base.get("n_peaks", 0)
        foc_peaks += foc.get("n_peaks", 0)
        verdict = add["missing_add_signal"].get("verdict")
        miss_tp += int(verdict == "TP")
        miss_fp += int(verdict == "FP")
        miss_fn += int(verdict == "FN")
        miss_tn += int(verdict == "TN")

        choc = rec_res["chocolate_count"]
        if choc.get("present") and choc.get("predicted_dark_blob_count") is not None:
            choc_known += 1
            choc_exact += int(choc.get("verdict") == "exact")
            choc_side += int(choc.get("verdict") in ("exact", "same_side_of_nominal"))
        rec = rec_res.get("recording")
        if rec in CHOCOLATE_TARGET_RECS:
            chocolate_targets[rec] = {
                "expected_count": choc.get("expected_count"),
                "predicted_dark_blob_count": choc.get("predicted_dark_blob_count"),
                "verdict": choc.get("verdict"),
                "status": choc.get("status"),
            }

        ocr = rec_res.get("microwave_ocr", {})
        if ocr.get("ready"):
            for step in ocr.get("steps", {}).values():
                if step.get("present"):
                    ocr_present += 1
                    ocr_usable += int(step.get("status") == "usable")

    return {
        "baseline_rgb_adds": {
            "total_expected_adds": base_exp,
            "total_matched_TP": base_tp,
            "total_peaks": base_peaks,
            "pooled_recall": rounded(base_tp / base_exp) if base_exp else None,
            "pooled_precision": rounded(base_tp / base_peaks) if base_peaks else None,
        },
        "focused_add_events": {
            "total_expected_adds": foc_exp,
            "total_matched_TP": foc_tp,
            "total_peaks": foc_peaks,
            "pooled_recall": rounded(foc_tp / foc_exp) if foc_exp else None,
            "pooled_precision": rounded(foc_tp / foc_peaks) if foc_peaks else None,
            "target_recall_ge_0_90": bool(foc_exp and (foc_tp / foc_exp) >= 0.90),
            "target_precision_ge_0_60": bool(foc_peaks and (foc_tp / foc_peaks) >= 0.60),
        },
        "missing_add_signal": {
            "TP": miss_tp,
            "FP": miss_fp,
            "FN": miss_fn,
            "TN": miss_tn,
        },
        "chocolate_dark_blob_count": {
            "known_predictions": choc_known,
            "exact": choc_exact,
            "same_side_of_nominal_or_exact": choc_side,
            "exact_accuracy": rounded(choc_exact / choc_known) if choc_known else None,
            "nominal_side_accuracy": rounded(choc_side / choc_known) if choc_known else None,
            "target_cases": chocolate_targets,
            "note": "dark-blob count is diagnostic only; object-model diagnostics are available with --models",
        },
        "microwave_ocr": {
            "present_steps": ocr_present,
            "usable_steps": ocr_usable,
            "usable_fraction": rounded(ocr_usable / ocr_present) if ocr_present else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="single recording id")
    ap.add_argument("--models", choices=["none", "groundingdino", "owlv2", "yoloworld"],
                    default="none")
    ap.add_argument("--model-max-frames", type=int, default=3)
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--ocr-step-s", type=float, default=2.0)
    ap.add_argument("--ocr-max-frames-per-step", type=int, default=12)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    recs = [args.only] if args.only else SHC_RECS
    object_probe = ObjectProbe(args.models)
    ocr_probe = OcrProbe(enabled=not args.skip_ocr)

    per = {}
    t0 = time.time()
    for rec in recs:
        res = {
            "recording": rec,
            "add_events": score_add_events(rec),
            "chocolate_count": score_chocolate_count(rec),
            "microwave_ocr": score_microwave_ocr(
                rec, ocr_probe, args.ocr_step_s, args.ocr_max_frames_per_step
            ),
        }
        if args.models != "none":
            res["object_model_diagnostics"] = object_model_diagnostics(
                rec, object_probe, args.model_max_frames
            )
        per[rec] = res

    out = {
        "probe": "cv_shc_criteria",
        "scope": {
            "included": ["microwave_display_ocr", "generic_add_events",
                         "chocolate_piece_count", "missing_add_signal"],
            "excluded": ["fill_milk", "mix", "cinnamon_vs_sugar_identity"],
            "video_dir": VID360,
            "gt_windows_for_offline_eval_only": True,
            "diagnostic_480_records": sorted(DIAGNOSTIC_480_RECS),
            "chocolate_target_records": sorted(CHOCOLATE_TARGET_RECS),
        },
        "config": {
            "sample_fps": SAMPLE_FPS,
            "mug_roi_frac": list(MUG_ROI),
            "tol_s": TOL_S,
            "focused_chi2_min": FOCUSED_CHI2_MIN,
            "focused_minsep_s": FOCUSED_MINSEP_S,
            "add_event_gate": "mug_roi_hsv_state_change_plus_30s_episode_separation",
            "chocolate_count_method": "low_confidence_dark_blob_proxy_in_mug_roi",
            "object_model": args.models,
            "ocr_enabled": not args.skip_ocr,
            "ocr_step_s": args.ocr_step_s,
            "ocr_max_frames_per_step": args.ocr_max_frames_per_step,
        },
        "summary": aggregate(per),
        "per_recording": per,
        "wall_s": rounded(time.time() - t0, 2),
    }

    print(json.dumps(out["summary"], indent=2))
    if not args.only:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(per[args.only], indent=2)[:6000])


if __name__ == "__main__":
    main()
