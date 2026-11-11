"""NIfTI filename handling and directory render planning."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


_NIFTI_SUFFIXES = (".nii.gz", ".nii")


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
