"""Pure/lazy functions over nibabel volumes."""

from __future__ import annotations

import os
from typing import cast

import nibabel as nib
import numpy as np
from nibabel.orientations import inv_ornt_aff, io_orientation, ornt_transform


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


def align_overlay_volume(
    base: nib.Nifti1Image, overlay: nib.Nifti1Image, *, name: str,
) -> nib.Nifti1Image:
    """Put an overlay on the base voxel grid using axis permutations and flips."""
    if len(overlay.shape) not in (3, 4):
        raise ValueError(f"{name} must be 3D or 4D.")
    if len(overlay.shape) == 4 and overlay.shape[3:] != base.shape[3:]:
        raise ValueError(f"{name} must be 3D or have the same number of frames as the 4D base image.")

    geometry_error = (
        f"{name} geometry must match the base image's spatial shape and voxel-to-world affine."
    )
    if not np.all(np.isfinite(base.affine)) or not np.all(np.isfinite(overlay.affine)):
        raise ValueError(geometry_error)
    base_orientation = io_orientation(base.affine)  # type: ignore[no-untyped-call]
    overlay_orientation = io_orientation(overlay.affine)  # type: ignore[no-untyped-call]
    if np.any(np.isnan(base_orientation)) or np.any(np.isnan(overlay_orientation)):
        raise ValueError(geometry_error)
    if base.shape[:3] == overlay.shape[:3] and np.allclose(
        base.affine, overlay.affine, rtol=0, atol=1e-4,
    ):
        return overlay

    transform = ornt_transform(overlay_orientation, base_orientation)  # type: ignore[no-untyped-call]
    shape = tuple(overlay.shape[axis] for axis in np.argsort(transform[:, 0]))
    affine = overlay.affine @ inv_ornt_aff(transform, overlay.shape[:3])  # type: ignore[no-untyped-call]
    if shape != base.shape[:3] or not np.allclose(base.affine, affine, rtol=0, atol=1e-4):
        raise ValueError(geometry_error)
    return cast(nib.Nifti1Image, overlay.as_reoriented(transform))  # type: ignore[no-untyped-call]
