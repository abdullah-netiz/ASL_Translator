#!/usr/bin/env python3
"""
Gate 0: Build manifest CSVs and verify CSV↔video match for each split.

Usage:
    python3 scripts/build_manifests.py --config configs/data.yaml --split all
    python3 scripts/build_manifests.py --config configs/data.yaml --split train

Output:
    data/manifests/{split}_manifest.csv   — columns: clip_id, video_path, sentence,
                                            duration_start, duration_end, split
    data/manifests/{split}_missing.txt    — one clip_id per line
    Prints Gate 0 summary table to stdout.
    Exits with code 1 if any processed split has > 10% missing.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional pandas import — fall back to stdlib csv module if unavailable
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _load_csv_pandas(path: str) -> list[dict]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict("records")


def _load_csv_stdlib(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def load_csv(path: str) -> list[dict]:
    if _HAS_PANDAS:
        return _load_csv_pandas(path)
    return _load_csv_stdlib(path)


# ---------------------------------------------------------------------------
# Resolve config paths
# ---------------------------------------------------------------------------

def _resolve(cfg: dict, root: Path) -> dict:
    """Resolve relative paths under root."""
    path_keys = {"raw_video_dirs", "translation_csvs", "landmark_dirs",
                 "manifest_dir", "tokenizer_dir", "data_root"}
    out = dict(cfg)
    for k, v in cfg.items():
        if k in path_keys:
            if isinstance(v, dict):
                out[k] = {kk: str(root / vv) for kk, vv in v.items()}
            elif isinstance(v, str):
                out[k] = str(root / v)
    return out


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_manifest_for_split(
    split: str,
    csv_path: str,
    video_dir: str,
    manifest_dir: str,
) -> dict:
    """
    Returns a stats dict:
      split, total_csv_rows, videos_found, videos_missing, missing_pct
    Writes manifest CSV and missing.txt.
    """
    csv_path_p = Path(csv_path)
    video_dir_p = Path(video_dir)
    manifest_dir_p = Path(manifest_dir)
    manifest_dir_p.mkdir(parents=True, exist_ok=True)

    if not csv_path_p.exists():
        print(f"WARNING: CSV not found for split '{split}': {csv_path_p} — skipping.",
              file=sys.stderr)
        return {}

    rows = load_csv(str(csv_path_p))
    total = len(rows)

    found_rows = []
    missing_clips = []

    for row in rows:
        sentence_name = row.get("SENTENCE_NAME", "").strip()
        if not sentence_name:
            missing_clips.append(("__EMPTY_SENTENCE_NAME__", ""))
            continue
        video_filename = sentence_name + ".mp4"
        video_path = video_dir_p / video_filename
        if video_path.exists():
            found_rows.append({
                "clip_id": sentence_name,
                "video_path": str(video_path),
                "sentence": row.get("SENTENCE", "").strip(),
                "duration_start": row.get("START", ""),
                "duration_end": row.get("END", ""),
                "split": split,
            })
        else:
            missing_clips.append((sentence_name, str(video_path)))

    # Write manifest CSV
    manifest_path = manifest_dir_p / f"{split}_manifest.csv"
    fieldnames = ["clip_id", "video_path", "sentence", "duration_start", "duration_end", "split"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(found_rows)

    # Write missing list
    missing_path = manifest_dir_p / f"{split}_missing.txt"
    with open(missing_path, "w", encoding="utf-8") as f:
        for clip_id, path in missing_clips:
            f.write(f"{clip_id}\t{path}\n")

    missing_pct = 100.0 * len(missing_clips) / total if total > 0 else 0.0

    return {
        "split": split,
        "total_csv_rows": total,
        "videos_found": len(found_rows),
        "videos_missing": len(missing_clips),
        "missing_pct": missing_pct,
        "manifest_path": str(manifest_path),
        "missing_path": str(missing_path),
    }


def print_gate0_table(stats_list: list[dict]) -> bool:
    """Print summary table. Returns True if all splits PASS."""
    print("\n" + "=" * 60)
    print("  GATE 0 — CSV ↔ VIDEO MATCH SUMMARY")
    print("=" * 60)
    header = f"{'split':<12} {'total_csv_rows':>14} {'videos_found':>13} {'videos_missing':>15} {'missing_pct':>11}"
    print(header)
    print("-" * 60)
    all_pass = True
    for s in stats_list:
        pct = s["missing_pct"]
        flag = "⚠ " if pct > 10.0 else "  "
        print(f"{flag}{s['split']:<10} {s['total_csv_rows']:>14} {s['videos_found']:>13} "
              f"{s['videos_missing']:>15} {pct:>10.1f}%")
        if pct > 10.0:
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("Gate 0 PASS — all processed splits within missing threshold.")
    else:
        print("Gate 0 FAIL — one or more splits exceed 10% missing videos.")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build How2Sign manifests and run Gate 0 check.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    # Load config (minimal inline load to avoid circular import)
    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install PyYAML", file=sys.stderr)
        return 1

    root = Path(args.config).resolve().parents[1]  # project root
    cfg = _resolve(cfg, root)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_stats = []

    for split in splits:
        csv_path = cfg["translation_csvs"][split]
        video_dir = cfg["raw_video_dirs"][split]
        manifest_dir = cfg["manifest_dir"]

        print(f"Processing split: {split}")
        print(f"  CSV:       {csv_path}")
        print(f"  Video dir: {video_dir}")

        stats = build_manifest_for_split(split, csv_path, video_dir, manifest_dir)
        if stats:
            all_stats.append(stats)
            print(f"  Manifest:  {stats['manifest_path']}")
            print(f"  Missing:   {stats['missing_path']}")

    if not all_stats:
        print("No splits were processed (no CSVs found).")
        return 0

    all_pass = print_gate0_table(all_stats)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
