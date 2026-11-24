"""Shared-PC guard: size worker counts from what is actually free right now."""
from __future__ import annotations
import os, subprocess


def free_ram_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def free_vram_gb() -> float:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        return float(out.split()[0]) / 1024
    except Exception:
        return 0.0


def idle_cores() -> float:
    return max(0.0, os.cpu_count() - os.getloadavg()[0])


def safe_workers(per_worker_gb: float, reserve_gb: float = 6.0, cap: int = 8, cpu_frac: float = 0.5) -> int:
    """Workers that fit in free RAM (keeping `reserve_gb` for other users / VS Code)
    and use at most `cpu_frac` of the currently idle cores."""
    by_ram = int((free_ram_gb() - reserve_gb) // per_worker_gb)
    by_cpu = int(idle_cores() * cpu_frac)
    return max(1, min(cap, by_ram, by_cpu))


if __name__ == "__main__":
    print(f"RAM free {free_ram_gb():.1f} GB | VRAM free {free_vram_gb():.1f} GB | idle cores {idle_cores():.1f} | "
          f"extract workers {safe_workers(1.3)} | loader workers {safe_workers(0.6, cap=4)}")
