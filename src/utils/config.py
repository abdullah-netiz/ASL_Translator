"""Config loader: load YAML, merge CLI overrides, resolve relative paths."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def merge_configs(*configs: dict) -> dict:
    """Left-merge: later configs override earlier ones (shallow)."""
    merged: dict = {}
    for c in configs:
        merged.update(c)
    return merged


def resolve_paths(cfg: dict, root: str | Path) -> dict:
    """Resolve relative paths in known path keys to absolute paths under root."""
    root = Path(root).resolve()
    path_keys = {
        "raw_video_dirs", "translation_csvs", "landmark_dirs",
        "manifest_dir", "tokenizer_dir", "data_root",
    }
    out = dict(cfg)
    for k, v in cfg.items():
        if k in path_keys:
            if isinstance(v, dict):
                out[k] = {kk: str(root / vv) for kk, vv in v.items()}
            elif isinstance(v, str):
                out[k] = str(root / v)
    return out


def build_argparser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config keys: --set fps_target=10 max_seq_len=128",
    )
    return p


def parse_overrides(overrides: list[str]) -> dict:
    out: dict = {}
    for item in overrides or []:
        k, _, v = item.partition("=")
        for cast in (int, float):
            try:
                v = cast(v)  # type: ignore[assignment]
                break
            except ValueError:
                pass
        out[k.strip()] = v
    return out
