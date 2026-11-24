"""Extract 1200-dim landmark features for WLASL isolated-sign videos.

Reuses the exact HolisticExtractor + normalize_sequence used for How2Sign so the
live browser/server pipeline keeps feature parity. Differences vs the How2Sign run:
  * time-based resampling to 15 fps (WLASL sources are 25/30 fps, mixed)
  * crop to the annotated signer bounding box (WLASL clips contain extra scenery)
Writes data/wlasl/landmarks/<video_id>.npy and data/wlasl/manifest.csv
"""
from __future__ import annotations
import csv, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import cv2, numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.extract.holistic import HolisticExtractor
from src.extract.normalize import normalize_sequence

TARGET_FPS = 15.0
RAW = ROOT / "data/archive/videos"
OUT = ROOT / "data/wlasl/landmarks"
LM_CFG = yaml.safe_load(open(ROOT / "configs/landmarks.yaml"))
_ex = None


def _init():
    global _ex
    warnings.filterwarnings("ignore")
    _ex = HolisticExtractor(static_image_mode=False, model_complexity=1)


def work(job):
    vid, path, bbox, f0, f1 = job
    out = OUT / f"{vid}.npy"
    if out.exists():
        return vid, True, int(np.load(out, mmap_mode="r").shape[0]), ""
    try:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = fps / TARGET_FPS
        frames, i = [], 0
        nxt = float(max(f0 - 1, 0))
        _ex.holistic.reset() if hasattr(_ex.holistic, "reset") else None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i + 1 < f0:            # official trim: frame_start is 1-based
                i += 1; continue
            if f1 > 0 and i + 1 > f1:
                break
            if i >= nxt - 1e-6:
                if bbox:
                    h, w = fr.shape[:2]
                    bx0, by0, bx1, by1 = bbox      # WLASL bbox: pixel x1,y1,x2,y2
                    mx, my = 0.10 * (bx1 - bx0), 0.10 * (by1 - by0)
                    x0, y0 = max(0, int(bx0 - mx)), max(0, int(by0 - my))
                    x1, y1 = min(w, int(bx1 + mx)), min(h, int(by1 + my))
                    if x1 - x0 > 32 and y1 - y0 > 32:
                        fr = fr[y0:y1, x0:x1]
                frames.append(_ex.process_frame(fr))
                nxt += step
            i += 1
        cap.release()
        if len(frames) < 4:
            return vid, False, 0, "too_short"
        arr = normalize_sequence(frames, LM_CFG, max_seq_len=96)
        np.save(out, arr)
        return vid, True, int(arr.shape[0]), ""
    except Exception as e:
        return vid, False, 0, repr(e)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, meta = [], {}
    for g in json.load(open(ROOT / "data/archive/WLASL_v0.3.json")):
        for ins in g["instances"]:
            vid = ins["video_id"]; p = RAW / f"{vid}.mp4"
            if not p.exists():
                continue
            jobs.append((vid, p, ins.get("bbox"), int(ins.get("frame_start", 1)), int(ins.get("frame_end", -1))))
            meta[vid] = (g["gloss"], ins["split"], ins.get("signer_id", -1))
    print("videos on disk:", len(jobs), flush=True)
    rows = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 20, initializer=_init) as pool:
        for k, (vid, ok, n, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            if ok:
                rows.append((vid, *meta[vid], n))
            elif err:
                print("FAIL", vid, err, flush=True)
            if k % 500 == 0:
                print(k, len(jobs), flush=True)
    with open(ROOT / "data/wlasl/manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "gloss", "split", "signer", "n_frames"]); w.writerows(sorted(rows))
    print("done", len(rows))


if __name__ == "__main__":
    main()
