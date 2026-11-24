"""NIfTI filename handling, render output naming, and directory render planning."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

from mrplot.slicing import Plane


_NIFTI_SUFFIXES = (".nii.gz", ".nii")
_MIN_FRAME_DIGITS = 4


class RenderJob(NamedTuple):
    nifti_file: str
    output_dir: str


def nifti_basename(path: str | os.PathLike[str]) -> str:
    """Return the filename with its NIfTI suffix removed."""
    filename = os.path.basename(path)
    for suffix in _NIFTI_SUFFIXES:
        if filename.lower().endswith(suffix):
            return filename[:-len(suffix)]
    return filename


def plane_output_stem(output_dir: str | os.PathLike[str], base_name: str, plane: Plane) -> str:
    """Path of a plane's output without its suffix: ``<output_dir>/<base_name>_<plane>``."""
    return os.path.join(output_dir, f"{base_name}_{plane.value}")


def frame_output_path(stem: str, index: int, frame_count: int) -> str:
    """PNG path for one volume of a time series: ``<stem>_vol-NNNN.png``.

    Indices are zero-padded to at least four digits, or more when the series needs them,
    so the files sort in volume order.
    """
    width = max(_MIN_FRAME_DIGITS, len(str(frame_count - 1)))
    return f"{stem}_vol-{index:0{width}d}.png"


class PlaneOutputs(NamedTuple):
    """One render's PNG output for a plane: a single image, or frames by volume index."""

    single: Path | None
    frames: Mapping[int, Path]


def find_plane_outputs(folder: str | os.PathLike[str], plane: Plane) -> PlaneOutputs:
    """Find the PNGs that one ``render`` wrote for ``plane`` in an output folder.

    Reads back the names written by :func:`plane_output_stem` and :func:`frame_output_path`.
    The folder must hold exactly one render for the plane: one ``*_<plane>.png``, or
    ``*_<plane>_vol-NNNN.png`` frames sharing one base name.
    """
    directory = Path(folder)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    singles = sorted(directory.glob(f"*_{plane.value}.png"))
    frame_name = re.compile(rf"(.*)_{plane.value}_vol-(\d+)\.png")
    stems: set[str] = set()
    frames: dict[int, Path] = {}
    for path in directory.iterdir():
        match = frame_name.fullmatch(path.name)
        if match:
            stems.add(match[1])
            frames[int(match[2])] = path

    pattern = f"*_{plane.value}.png"
    if len(singles) > 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {', '.join(p.name for p in singles)}.")
    if singles and frames:
        raise ValueError(
            f"{directory} has both {singles[0].name} and *_{plane.value}_vol-NNNN.png frames; "
            "keep one render per folder."
        )
    if len(stems) > 1:
        raise ValueError(f"Frames from several renders in {directory}: {', '.join(sorted(stems))}.")
    if not singles and not frames:
        raise ValueError(
            f"Expected {pattern} or *_{plane.value}_vol-NNNN.png frames in {directory}, found none."
        )
    return PlaneOutputs(singles[0] if singles else None, dict(sorted(frames.items())))


def _validate_directory_roots(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise ValueError(f"input is not a directory: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("input and output directories must differ")


def _iter_file_destinations(
    input_dir: Path, output_dir: Path, *, suffixes: tuple[str, ...],
) -> Iterator[tuple[Path, Path]]:
    """Yield files matching the supplied lowercase suffixes and their output directories."""
    walk_errors: list[OSError] = []
    for root, directories, filenames in os.walk(input_dir, onerror=walk_errors.append):
        source_dir = Path(root)
        directories[:] = sorted(
            name for name in directories if source_dir.joinpath(name) != output_dir
        )
        target_dir = output_dir.joinpath(source_dir.relative_to(input_dir))
        for filename in sorted(filenames):
            if not filename.lower().endswith(suffixes):
                continue
            source = source_dir.joinpath(filename)
            if source.is_symlink() or not source.is_file():
                continue
            yield source, target_dir
    if walk_errors:
        raise walk_errors[0]


def directory_jobs(
    input_path: str | os.PathLike[str], output_path: str | os.PathLike[str]
) -> Iterator[RenderJob]:
    """Yield render jobs, rejecting conflicting destinations."""
    input_dir = Path(input_path).resolve()
    output_dir = Path(output_path).resolve()
    _validate_directory_roots(input_dir, output_dir)

    sources_by_destination: dict[tuple[Path, str], Path] = {}
    for source, target_dir in _iter_file_destinations(
        input_dir, output_dir, suffixes=_NIFTI_SUFFIXES,
    ):
        destination = (target_dir, nifti_basename(source.name))
        previous = sources_by_destination.get(destination)
        if previous is not None:
            raise ValueError(f"output name collision between {previous} and {source}")
        sources_by_destination[destination] = source
        yield RenderJob(nifti_file=str(source), output_dir=str(target_dir))
