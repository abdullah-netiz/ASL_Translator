"""Record your own sign samples from a webcam for fine-tuning (closes the WLASL -> your-camera gap).

  python scripts/record_signs.py hello thank_you help water --n 8 [--cam 0 | --cam http://PHONE:8080/video]

SPACE = start/stop one take, Q = quit, S = skip word. Clips are saved in the exact feature
format used for training (15 fps, 1200-dim) to data/wlasl/custom/<word>__<k>.npy
"""
import argparse, sys, time, warnings
from pathlib import Path
import cv2, numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.extract.holistic import HolisticExtractor
from src.extract.normalize import normalize_sequence

ap = argparse.ArgumentParser(); ap.add_argument("words", nargs="+"); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--cam", default="0"); a = ap.parse_args()
warnings.filterwarnings("ignore")
cfg = yaml.safe_load(open(ROOT / "configs/landmarks.yaml")); out = ROOT / "data/wlasl/custom"; out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(int(a.cam) if a.cam.isdigit() else a.cam); ex = HolisticExtractor(); STEP = 1 / 15
for w in a.words:
    k = len(list(out.glob(f"{w}__*.npy")))
    while k < a.n:
        rec, frames, nxt, t0 = False, [], 0.0, 0.0
        while True:
            ok, fr = cap.read()
            if not ok: sys.exit("camera read failed")
            vis = fr.copy()
            cv2.putText(vis, f"{w}  take {k+1}/{a.n}  {'REC' if rec else 'SPACE=start'}", (10, 30), 0, 0.9, (0, 0, 255) if rec else (0, 255, 0), 2)
            cv2.imshow("record", vis); key = cv2.waitKey(1) & 0xFF
            if rec and time.time() - t0 >= nxt:
                frames.append(ex.process_frame(fr)); nxt += STEP
            if key == ord("q"): sys.exit()
            if key == ord("s"): k = a.n; break
            if key == 32:
                if not rec: rec, frames, t0, nxt = True, [], time.time(), 0.0
                else:
                    rec = False
                    if len(frames) >= 6:
                        np.save(out / f"{w}__{k:02d}.npy", normalize_sequence(frames, cfg, max_seq_len=96)); k += 1; print(w, k)
                    break
print("done")
