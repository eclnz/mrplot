"""I/O boundary: writing rendered frames to disk as png/mp4."""

from __future__ import annotations

import os
from typing import Any, Iterable

import imageio
from numpy.typing import NDArray


def validate_fps(fps: int) -> None:
    if fps <= 0:
        raise ValueError("Frames per second must be positive.")


def write_png(path: str, frame: NDArray[Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.imwrite(path, frame)
    return path


def write_mp4_frames(
    path: str, frames: Iterable[NDArray[Any]], fps: int, *, macro_block_size: int = 16,
) -> str:
    validate_fps(fps)
    if macro_block_size <= 0:
        raise ValueError("macro_block_size must be positive.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", macro_block_size=macro_block_size,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return path
