"""
CLI: extract MediaPipe Holistic landmarks from How2Sign video clips in parallel.

Usage:
    python -m src.extract.run_extraction \
        --config configs/data.yaml \
        --config-lm configs/landmarks.yaml \
        --split val

    python -m src.extract.run_extraction \
        --config configs/data.yaml \
        --config-lm configs/landmarks.yaml \
        --split all \
        --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

# Suppress protobuf and TF C++ warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_paths
from src.utils.logging import get_logger
from src.extract.holistic import HolisticExtractor
from src.extract.normalize import normalize_sequence

logger = get_logger("extract.run")

_worker_extractor: HolisticExtractor | None = None


def _init_worker() -> None:
    """Initialize a single persistent HolisticExtractor instance per worker process."""
    global _worker_extractor
    warnings.filterwarnings("ignore")
    try:
        _worker_extractor = HolisticExtractor(static_image_mode=False, model_complexity=1)
    except Exception as e:
        logger.error(f"Failed to initialize HolisticExtractor in worker: {e}")
        _worker_extractor = None


def _load_manifest(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_one(args: tuple) -> dict:
    """
    Worker function executed in pool process.
    Reuses persistent _worker_extractor.
    """
    global _worker_extractor
    clip_id, video_path, out_path, data_cfg, lm_cfg = args
    result: dict = {
        "clip_id": clip_id,
        "out_path": out_path,
        "success": False,
        "error": None,
        "n_frames": 0,
        "lh_rate": 0.0,
        "rh_rate": 0.0,
        "truncated": False,
    }
    try:
        stride = int(data_cfg.get("stride", 2))
        max_seq_len = int(data_cfg.get("max_seq_len", 256))

        if _worker_extractor is None:
            _init_worker()

        if _worker_extractor is None:
            result["error"] = "extractor_init_failed"
            return result

        frames = _worker_extractor.extract_video(video_path, stride=stride)

        if len(frames) == 0:
            result["error"] = "no_frames"
            return result

        n = len(frames)
        lh_detected = sum(1 for f in frames if f["left_hand_detected"])
        rh_detected = sum(1 for f in frames if f["right_hand_detected"])
        result["lh_rate"] = lh_detected / n
        result["rh_rate"] = rh_detected / n
        result["truncated"] = n > max_seq_len

        arr = normalize_sequence(frames, lm_cfg, max_seq_len=max_seq_len)
        result["n_frames"] = int(arr.shape[0])

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
        result["success"] = True

    except Exception:
        result["error"] = traceback.format_exc()

    return result


def _build_jobs(split: str, data_cfg: dict, lm_cfg: dict, limit: int | None = None) -> list[tuple]:
    manifest_path = Path(data_cfg["manifest_dir"]) / f"{split}_manifest.csv"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found for split '{split}': {manifest_path}")
        return []

    manifest = _load_manifest(manifest_path)
    lm_dir = Path(data_cfg["landmark_dirs"][split])
    lm_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for row in manifest:
        clip_id = row["clip_id"]
        video_path = row["video_path"]
        out_path = str(lm_dir / f"{clip_id}.npy")
        if Path(out_path).exists():
            continue  # resume-safe skip
        jobs.append((clip_id, video_path, out_path, data_cfg, lm_cfg))
        if limit and len(jobs) >= limit:
            break
    return jobs


def _run_split(
    split: str,
    data_cfg: dict,
    lm_cfg: dict,
    n_workers: int,
    limit: int | None = None,
) -> dict:
    jobs = _build_jobs(split, data_cfg, lm_cfg, limit=limit)
    logger.info(f"[{split}] {len(jobs)} clips queued for extraction")

    if not jobs:
        logger.info(f"[{split}] Nothing to do (all extracted or no manifest).")
        return {}

    n_workers = min(n_workers, len(jobs), os.cpu_count() or 1)
    results: list[dict] = []

    # Use multiprocessing fork or spawn (fork is fast on Linux)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(_extract_one, jobs, chunksize=8)):
            results.append(res)
            if (i + 1) % 200 == 0 or (i + 1) == len(jobs):
                ok = sum(1 for r in results if r["success"])
                logger.info(f"[{split}] {i+1}/{len(jobs)} clips processed | success={ok}")

    succeeded = [r for r in results if r["success"]]
    failed    = [r for r in results if not r["success"]]
    lh_rates  = [r["lh_rate"] for r in succeeded]
    rh_rates  = [r["rh_rate"] for r in succeeded]
    lengths   = [r["n_frames"] for r in succeeded]
    truncated = sum(1 for r in results if r.get("truncated"))

    stats = {
        "split": split,
        "total_jobs": len(jobs),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "fail_rate_pct": 100.0 * len(failed) / len(jobs) if jobs else 0.0,
        "truncated": truncated,
        "lh_rate_mean": float(np.mean(lh_rates)) if lh_rates else 0.0,
        "rh_rate_mean": float(np.mean(rh_rates)) if rh_rates else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_median": float(np.median(lengths)) if lengths else 0.0,
        "length_max": int(np.max(lengths)) if lengths else 0,
        "length_p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
    }

    # Save or update per-clip QC log
    qc_path = Path(data_cfg["manifest_dir"]) / f"{split}_extraction_qc.json"
    existing = []
    if qc_path.exists():
        try:
            with open(qc_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    combined = {r["clip_id"]: r for r in existing}
    for r in results:
        combined[r["clip_id"]] = r
    with open(qc_path, "w") as f:
        json.dump(list(combined.values()), f, indent=2)
    logger.info(f"[{split}] QC log saved → {qc_path}")

    if failed:
        fail_path = Path(data_cfg["manifest_dir"]) / f"{split}_failed.txt"
        with open(fail_path, "a") as f:
            for r in failed:
                f.write(f"{r['clip_id']}\t{r['error']}\n")
        logger.warning(f"[{split}] {len(failed)} failures logged to {fail_path}")

    return stats


def _print_stats(s: dict) -> None:
    print(f"\n── Extraction stats: {s['split']} ──")
    print(f"  total={s['total_jobs']}  succeeded={s['succeeded']}  "
          f"failed={s['failed']}  fail%={s['fail_rate_pct']:.1f}%  "
          f"truncated={s['truncated']}")
    print(f"  lh_det={s['lh_rate_mean']:.3f}  rh_det={s['rh_rate_mean']:.3f}")
    print(f"  len: mean={s['length_mean']:.1f}  median={s['length_median']:.1f}"
          f"  p95={s['length_p95']:.1f}  max={s['length_max']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediaPipe landmark extraction")
    parser.add_argument("--config",    required=True, help="Path to configs/data.yaml")
    parser.add_argument("--config-lm", required=True, help="Path to configs/landmarks.yaml")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips to extract (for testing)")
    args = parser.parse_args()

    root = Path(args.config).resolve().parents[1]
    data_cfg = resolve_paths(load_config(args.config), root)
    lm_cfg   = load_config(args.config_lm)

    n_workers = args.workers or int(data_cfg.get("num_workers", 16))
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_stats = []
    for split in splits:
        stats = _run_split(split, data_cfg, lm_cfg, n_workers, limit=args.limit)
        if stats:
            all_stats.append(stats)
            _print_stats(stats)

    print("\n" + "=" * 60)
    print("GATE 1 CHECK")
    print("=" * 60)
    any_fail = False
    for s in all_stats:
        status = "PASS ✓" if s["fail_rate_pct"] < 5.0 else "FAIL ✗"
        if s["fail_rate_pct"] >= 5.0:
            any_fail = True
        print(f"  [{s['split']}] fail={s['fail_rate_pct']:.1f}%  "
              f"lh_det={s['lh_rate_mean']:.3f}  "
              f"len_p95={s['length_p95']:.0f}  → {status}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
