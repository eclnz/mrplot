import numpy as np
import pytest

from tests.phantoms import box_mask, ellipsoid, moving_sphere, save_phantom


# Voxel axes 0, 1, and 2 map to -S, -R, and +A respectively.
_PERMUTED_AFFINE = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


@pytest.fixture
def sample_3d_nifti(tmp_path):
    return save_phantom(tmp_path, "test3d", ellipsoid())


@pytest.fixture
def sample_4d_nifti(tmp_path):
    return save_phantom(tmp_path, "test4d", moving_sphere((50, 50, 50), frames=10))


@pytest.fixture
def sample_5d_nifti(tmp_path):
    return save_phantom(tmp_path, "test5d", np.zeros((8, 8, 8, 3, 2), dtype=np.uint8))


@pytest.fixture
def sample_mask(tmp_path):
    return save_phantom(tmp_path, "mask", box_mask())


@pytest.fixture
def sample_3d_nifti_flipped_affine(tmp_path):
    """A 3D volume whose affine permutes and flips axes relative to RAS+,
    to exercise orientation-aware (non-canonical) slicing."""
    return save_phantom(tmp_path, "test3d_flipped", ellipsoid(), _PERMUTED_AFFINE)


@pytest.fixture
def sample_mask_flipped_affine(tmp_path):
    return save_phantom(tmp_path, "mask_flipped", box_mask(), _PERMUTED_AFFINE)
