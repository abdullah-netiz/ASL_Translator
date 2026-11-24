"""Extract 1200-dim landmarks for ASL Citizen directly from ASL_Citizen.zip (no full unzip).

Same HolisticExtractor + normalize_sequence + 15 fps resampling as WLASL / the live camera.
Resumable; RAM-guarded (workers pause when free RAM < 4 GB). Worker count from free resources.
  python scripts/aslcitizen_extract.py [workers]
"""
from __future__ import annotations
import csv, io, os, sys, tempfile, time, warnings, zipfile
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.extract.holistic import HolisticExtractor
from src.extract.normalize import normalize_sequence
from src.sign.resources import free_ram_gb, safe_workers

ZIP = ROOT / "data/aslcitizen/ASL_Citizen.zip"
OUT = ROOT / "data/aslcitizen/landmarks"
TMP = ROOT / "data/aslcitizen/tmp"
LM_CFG = yaml.safe_load(open(ROOT / "configs/landmarks.yaml"))
TARGET_FPS = 15.0
_ex = _zf = None


def _init():
    global _ex, _zf
    warnings.filterwarnings("ignore")
    _ex = HolisticExtractor(static_image_mode=False, model_complexity=1)
    _zf = zipfile.ZipFile(ZIP)


def work(job):
    vid, member = job
    out = OUT / f"{vid}.npy"
    if out.exists():
        return vid, True, int(np.load(out, mmap_mode="r").shape[0]), ""
    while free_ram_gb() < 4.0:          # shared-PC guard
        time.sleep(5)
    tmp = TMP / f"{os.getpid()}.mp4"
    try:
        with _zf.open(member) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        cap = cv2.VideoCapture(str(tmp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps < 5 or fps > 120:
            fps = 30.0
        step = fps / TARGET_FPS; frames, i, nxt = [], 0, 0.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i >= nxt - 1e-6:
                h, w = fr.shape[:2]
                if max(h, w) > 960:                       # speed: MediaPipe works at ~256px internally
                    s = 960 / max(h, w); fr = cv2.resize(fr, (int(w * s), int(h * s)))
                frames.append(_ex.process_frame(fr)); nxt += step
            i += 1
        cap.release()
        if len(frames) < 4:
            return vid, False, 0, "too_short"
        arr = normalize_sequence(frames, LM_CFG, max_seq_len=96)
        np.save(out, arr)
        return vid, True, int(arr.shape[0]), ""
    except Exception as e:
        return vid, False, 0, repr(e)[:200]
    finally:
        tmp.unlink(missing_ok=True)


def read_splits(zf: zipfile.ZipFile):
    """Find split CSVs inside the zip; tolerate column-name variants."""
    rows = []
    names = zf.namelist()
    vids = {Path(n).name: n for n in names if n.lower().endswith(".mp4")}
    for n in names:
        base = Path(n).name.lower()
        if not base.endswith(".csv"):
            continue
        split = next((s for s in ("train", "val", "test") if base.startswith(s)), None)
        if split is None:
            continue
        r = csv.DictReader(io.TextIOWrapper(zf.open(n), encoding="utf-8-sig"))
        for row in r:
            low = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            f = low.get("video file") or low.get("video_file") or low.get("filename")
            g = low.get("gloss"); p = low.get("participant id") or low.get("participant_id") or ""
            if f and g and f in vids:
                # English word from the file name ('123-NOT MIND.mp4' -> 'NOT MIND'); CSV gloss drops spaces
                word = Path(f).stem.split("-", 1)[-1] if "-" in Path(f).stem else g
                rows.append((Path(f).stem.split("-", 1)[0], vids[f], word, split, p, g))
    # canonical English word per official gloss = most common file-name spelling
    from collections import Counter
    by = {}
    for r in rows:
        by.setdefault(r[5], Counter())[r[2]] += 1
    canon = {g: c.most_common(1)[0][0] for g, c in by.items()}
    return [(r[0], r[1], canon[r[5]], r[3], r[4]) for r in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    rows = read_splits(zipfile.ZipFile(ZIP))
    print("rows in split csvs:", len(rows), flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else safe_workers(1.3, cap=8)
    print("workers:", nw, flush=True)
    meta = {r[0]: r for r in rows}; done = []
    with Pool(nw, initializer=_init) as pool:
        for k, (vid, ok, n, err) in enumerate(pool.imap_unordered(work, [(r[0], r[1]) for r in rows], chunksize=8)):
            if ok:
                done.append((vid, meta[vid][2], meta[vid][3], meta[vid][4], n))
            elif err:
                print("FAIL", vid, err, flush=True)
            if k % 1000 == 0:
                print(time.strftime("%H:%M:%S"), k, len(rows), flush=True)
    with open(ROOT / "data/aslcitizen/manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "gloss", "split", "signer", "n_frames"]); w.writerows(sorted(done))
    print("done", len(done), flush=True)


if __name__ == "__main__":
    main()
