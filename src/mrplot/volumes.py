"""Pure/lazy functions over nibabel volumes."""

from __future__ import annotations

import os

import nibabel as nib


def validate_volume_path(path: str | os.PathLike[str]) -> None:
    """Require a path to an existing volume file."""
    if not os.path.isfile(path):
        raise ValueError(f"no such file: {path}")


def load_volume(path: str | os.PathLike[str]) -> nib.Nifti1Image:
    """Load a NIfTI file. Reads the header only - no pixel data is touched."""
    image = nib.load(os.fspath(path))
    if not isinstance(image, nib.Nifti1Image):
        raise ValueError(f"{path} is not a NIfTI file (got {type(image).__name__}).")
    return image


def load_render_volume(path: str | os.PathLike[str]) -> nib.Nifti1Image:
    """Load a scan header and require the supported 3D or 4D input shape."""
    validate_volume_path(path)
    image = load_volume(path)
    ndim = len(image.shape)
    if ndim not in (3, 4):
        raise ValueError(
            f"{path}: unsupported data dimensionality: {ndim}D. "
            "Only 3D and 4D volumes are supported."
        )
    return image
