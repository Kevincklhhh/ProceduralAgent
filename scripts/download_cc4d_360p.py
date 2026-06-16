#!/usr/bin/env python3
"""Bulk-download CC4D GoPro 360p videos and extract 16k/48k mono wavs.

- Videos  -> data/videos_360p/{rec}.mp4   (naming matches videos_480p/)
- Wavs    -> data/audio/{rec}_16k.wav, {rec}_48k.wav (skips existing,
             e.g. the six activity-8 wavs already derived from 4K)

Only uses the `gopro_360p` link; recordings without one are logged, NOT
silently substituted with the HoloLens PV video (unlike the official
downloader). 360p audio was verified identical to the 4K track on 8_16
(see docs/DETECTOR_FEASIBILITY.md).

Usage: python3 scripts/download_cc4d_360p.py [--limit N] [--workers N]
Idempotent: safe to re-run; finished files are skipped.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINKS = ROOT / 'data/cc4d/downloader/metadata/download_links.json'
VIDEO_DIR = ROOT / 'data/videos_360p'
AUDIO_DIR = ROOT / 'data/audio'

MIN_VIDEO_BYTES = 1_000_000  # smaller than any real recording -> treat as partial


def log(msg):
    print(msg, flush=True)


def has_audio_stream(path):
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, timeout=60)
        return bool(out.stdout.strip())
    except Exception:
        return False


def download(rec, url):
    dest = VIDEO_DIR / f'{rec}.mp4'
    if dest.exists() and dest.stat().st_size >= MIN_VIDEO_BYTES:
        return 'cached'
    # Download to a temp file in the same dir, then atomic-rename.
    with tempfile.NamedTemporaryFile(dir=VIDEO_DIR, suffix='.part', delete=False) as tf:
        tmp = Path(tf.name)
    try:
        subprocess.run(
            ['curl', '-sL', '--fail', '--retry', '3', '--retry-delay', '5',
             '--connect-timeout', '30', '-o', str(tmp), url],
            check=True, timeout=1800)
        if tmp.stat().st_size < MIN_VIDEO_BYTES:
            raise RuntimeError(f'suspiciously small file ({tmp.stat().st_size} bytes)')
        if not has_audio_stream(tmp):
            raise RuntimeError('downloaded mp4 has NO audio stream')
        tmp.chmod(0o644)
        tmp.rename(dest)
        return 'downloaded'
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def extract_wavs(rec):
    src = VIDEO_DIR / f'{rec}.mp4'
    made = []
    for rate, tag in ((16000, '16k'), (48000, '48k')):
        dest = AUDIO_DIR / f'{rec}_{tag}.wav'
        if dest.exists() and dest.stat().st_size > 0:
            continue
        tmp = dest.with_suffix('.wav.part')
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-i', str(src),
             '-vn', '-ac', '1', '-ar', str(rate), '-f', 'wav', str(tmp)],
            check=True, timeout=600)
        tmp.rename(dest)
        made.append(tag)
    return made


def process(rec, url):
    status = download(rec, url)
    wavs = extract_wavs(rec)
    return status, wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='process only first N recordings')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    links = json.loads(LINKS.read_text())
    recs = sorted(links.keys(), key=lambda r: tuple(map(int, r.split('_'))))

    no_link = [r for r in recs if not links[r].get('gopro_360p')]
    todo = [r for r in recs if links[r].get('gopro_360p')]
    if args.limit:
        todo = todo[:args.limit]

    log(f'{len(todo)} recordings to process, {len(no_link)} with no gopro_360p link: {no_link}')

    counts = {'downloaded': 0, 'cached': 0, 'failed': 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, r, links[r]['gopro_360p']): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            try:
                status, wavs = fut.result()
                counts[status] += 1
                log(f'[{i}/{len(todo)}] {rec}: {status}'
                    + (f', wavs: {"+".join(wavs)}' if wavs else ', wavs: cached'))
            except Exception as e:
                counts['failed'] += 1
                failed.append(rec)
                log(f'[{i}/{len(todo)}] {rec}: FAILED — {e}')

    log(f'\nDone. downloaded={counts["downloaded"]} cached={counts["cached"]} '
        f'failed={counts["failed"]}')
    if no_link:
        log(f'No 360p link (need 4K or HoloLens fallback, handle manually): {no_link}')
    if failed:
        log(f'Failed (re-run to retry): {sorted(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
