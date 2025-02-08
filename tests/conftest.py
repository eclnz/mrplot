import pytest # type: ignore
import numpy as np # type: ignore
import nibabel as nib # type: ignore
import os


# Fixtures for different dimensionalities
@pytest.fixture
def sample_3d_nifti(tmp_path):
    data = np.random.rand(50, 50, 50)  # 3D volume
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(tmp_path, "test3d.nii.gz")
    img.to_filename(path)
    return path

@pytest.fixture
def sample_4d_nifti(tmp_path):
    data = np.random.rand(50, 50, 50, 10)  # 3D + time
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(tmp_path, "test4d.nii.gz")
    img.to_filename(path)
    return path

@pytest.fixture
def sample_5d_nifti(tmp_path):
    data = np.random.rand(50, 50, 50, 3, 10)  # 3D + time + echo
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(tmp_path, "test5d.nii.gz")
    img.to_filename(path)
    return path

# Mask and underlay fixtures
@pytest.fixture
def sample_mask(tmp_path):
    data = np.zeros((50, 50, 50), dtype=np.int8)
    data[20:30, 20:30, 20:30] = 1  # Cube mask
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(tmp_path, "mask.nii.gz")
    img.to_filename(path)
    return path

@pytest.fixture
def sample_underlay(tmp_path):
    data = np.random.rand(50, 50, 50)  # 3D underlay
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(tmp_path, "underlay.nii.gz")
    img.to_filename(path)
    return path