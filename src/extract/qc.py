"""
QC report generator: reads per-clip extraction JSON logs and writes a markdown summary.

Usage:
    python -m src.extract.qc \\
        --config configs/data.yaml \\
        --out reports/phase1_extraction_qc.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_paths
from src.utils.logging import get_logger

logger = get_logger("extract.qc")

# Clips with average hand detection rate below this are flagged
MIN_HAND_RATE = 0.30


def _load_qc_log(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _compute_stats(records: list[dict], split: str) -> dict:
    succeeded = [r for r in records if r.get("success")]
    failed    = [r for r in records if not r.get("success")]
    lh_rates  = [r["lh_rate"] for r in succeeded]
    rh_rates  = [r["rh_rate"] for r in succeeded]
    lengths   = [r["n_frames"] for r in succeeded]
    avg_rates = [(r["lh_rate"] + r["rh_rate"]) / 2 for r in succeeded]
    flagged   = [r for r, avg in zip(succeeded, avg_rates) if avg < MIN_HAND_RATE]

    return {
        "split": split,
        "total": len(records),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "fail_pct": 100.0 * len(failed) / len(records) if records else 0.0,
        "flagged_low_hand": len(flagged),
        "flagged_pct": 100.0 * len(flagged) / len(succeeded) if succeeded else 0.0,
        "lh_rate_mean": float(np.mean(lh_rates)) if lh_rates else 0.0,
        "rh_rate_mean": float(np.mean(rh_rates)) if rh_rates else 0.0,
        "len_mean": float(np.mean(lengths)) if lengths else 0.0,
        "len_median": float(np.median(lengths)) if lengths else 0.0,
        "len_p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "len_max": int(np.max(lengths)) if lengths else 0,
        "truncated": sum(1 for r in records if r.get("truncated")),
    }


def _stats_to_md(s: dict) -> str:
    gate = "✅ PASS" if s["fail_pct"] < 5.0 else "❌ FAIL"
    return f"""
## Split: {s['split']}

| Metric | Value |
|--------|-------|
| Total clips | {s['total']} |
| Succeeded | {s['succeeded']} |
| Failed | {s['failed']} ({s['fail_pct']:.1f}%) |
| Flagged (avg hand det < {MIN_HAND_RATE:.0%}) | {s['flagged_low_hand']} ({s['flagged_pct']:.1f}%) |
| Left hand detection rate (mean) | {s['lh_rate_mean']:.3f} |
| Right hand detection rate (mean) | {s['rh_rate_mean']:.3f} |
| Seq length mean | {s['len_mean']:.1f} frames |
| Seq length median | {s['len_median']:.1f} frames |
| Seq length p95 | {s['len_p95']:.1f} frames |
| Seq length max | {s['len_max']} frames |
| Truncated (> max_seq_len) | {s['truncated']} |

**Gate 1 status:** {gate} (fail rate < 5%)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 QC markdown report")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument("--out", default="reports/phase1_extraction_qc.md",
                        help="Output markdown path")
    args = parser.parse_args()

    root = Path(args.config).resolve().parents[1]
    data_cfg = resolve_paths(load_config(args.config), root)
    manifest_dir = Path(data_cfg["manifest_dir"])

    cfg_stride   = data_cfg.get("stride", 2)
    cfg_max_len  = data_cfg.get("max_seq_len", 256)

    md_parts = [
        "# Phase 1 — Landmark Extraction QC Report\n",
        f"**MediaPipe version:** 0.10.14  |  "
        f"**stride:** {cfg_stride}  |  "
        f"**max_seq_len:** {cfg_max_len}\n",
    ]

    any_fail = False
    for split in ["train", "val", "test"]:
        qc_log = manifest_dir / f"{split}_extraction_qc.json"
        if not qc_log.exists():
            md_parts.append(
                f"\n## Split: {split}\n\n"
                "_No QC log found — extraction not run yet._\n"
            )
            continue
        records = _load_qc_log(qc_log)
        s = _compute_stats(records, split)
        md_parts.append(_stats_to_md(s))
        logger.info(
            f"[{split}] fail={s['fail_pct']:.1f}%  "
            f"lh={s['lh_rate_mean']:.3f}  "
            f"len_p95={s['len_p95']:.0f}"
        )
        if s["fail_pct"] >= 5.0:
            any_fail = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(md_parts)
    out_path.write_text(content)
    print(f"\nQC report written to: {out_path}")
    print(content)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
