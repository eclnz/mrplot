#!/usr/bin/env python3
"""Benchmark `mrplot render` wall time and peak RSS across volume sizes.

Each case runs in its own subprocess (via `/usr/bin/time -l`) so peak
memory is measured per-call, not accumulated across the whole script. If
slicing is actually lazy, both time and memory should stay roughly flat as
volume size grows, since only three 2D(+time) planes are ever read.

Usage: .venv/bin/python scripts/benchmark.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np

MRPLOT = str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "mrplot")

CASES: dict[str, tuple[int, ...]] = {
    "64^3 (3D)": (64, 64, 64),
    "128^3 (3D)": (128, 128, 128),
    "256^3 (3D)": (256, 256, 256),
    "384^3 (3D)": (384, 384, 384),
    "128^3 x10 (4D)": (128, 128, 128, 10),
    "128^3 x50 (4D)": (128, 128, 128, 50),
    "128^3 x200 (4D)": (128, 128, 128, 200),
}


class Result(NamedTuple):
    label: str
    file_size_mb: float
    wall_seconds: float
    peak_rss_mb: float


def _make_volume(path: Path, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32)
    nib.Nifti1Image(data, affine=np.eye(4)).to_filename(str(path))


def _run_once(nifti_path: Path, output_dir: Path) -> tuple[float, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-l", MRPLOT, "render", str(nifti_path), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"mrplot render failed:\n{proc.stderr}")

    match = re.search(r"(\d+)\s+peak memory footprint", proc.stderr)
    if not match:
        raise RuntimeError(f"could not parse peak memory from:\n{proc.stderr}")
    peak_rss_mb = int(match.group(1)) / (1024 * 1024)
    return wall_seconds, peak_rss_mb


def main() -> None:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for label, shape in CASES.items():
            nifti_path = tmp_path / f"{label.replace(' ', '_').replace('^', '')}.nii.gz"
            _make_volume(nifti_path, shape)
            file_size_mb = nifti_path.stat().st_size / (1024 * 1024)

            output_dir = tmp_path / "out"
            wall_seconds, peak_rss_mb = _run_once(nifti_path, output_dir)
            results.append(Result(label, file_size_mb, wall_seconds, peak_rss_mb))
            nifti_path.unlink()

    header = f"{'case':<20}{'file size (MB)':>16}{'wall time (s)':>16}{'peak RSS (MB)':>16}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.label:<20}{r.file_size_mb:>16.1f}{r.wall_seconds:>16.2f}{r.peak_rss_mb:>16.1f}")


if __name__ == "__main__":
    sys.exit(main())
