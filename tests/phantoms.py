"""Deterministic geometric volumes generated locally for the test suite."""

from pathlib import Path

import nibabel as nib
import numpy as np


def save_phantom(directory: Path, name: str, data, affine=None) -> Path:
    path = directory / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)
    return path


def ellipsoid(shape=(50, 50, 50)):
    x, y, z = np.indices(shape, sparse=True, dtype=np.float32)
    x = (x - 0.48 * shape[0]) / (0.40 * shape[0])
    y = (y - 0.52 * shape[1]) / (0.35 * shape[1])
    z = (z - 0.49 * shape[2]) / (0.42 * shape[2])
    return np.maximum(0, 0.9 - x**2 - y**2 - z**2)


def box_mask(shape=(50, 50, 50), half_width=5):
    bounds = tuple(slice(size // 2 - half_width, size // 2 + half_width) for size in shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[bounds] = 1
    return mask


def moving_sphere(shape=(24, 24, 24), frames=2):
    x, y, z = np.indices(shape, sparse=True, dtype=np.float32)
    centers = np.linspace(0.3 * shape[0], 0.7 * shape[0], frames)
    radius = 0.16 * min(shape)
    distance = (x[..., None] - centers) ** 2 + (
        (y - shape[1] // 2) ** 2 + (z - shape[2] // 2) ** 2
    )[..., None]
    return (distance <= radius**2).astype(np.float32)
