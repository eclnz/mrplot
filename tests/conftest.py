import numpy as np
import nibabel as nib
import os
from pathlib import Path
import pytest

def create_nifti_3d(path: Path, shape=(50, 50, 50), mask=False):
    """Create a 3D NIfTI file at specified path. If mask is True, generates a cube mask."""
    if mask:
        data = np.zeros(shape, dtype=np.int8)
        center = np.round(np.array(shape) / 2).astype(int)  # Round the center coordinates
        half_size = 5  # Size of the cube mask
        data[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size, center[2]-half_size:center[2]+half_size] = 1  # Cube mask
    else:
        data = np.random.rand(*shape)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.to_filename(str(path))
    return path

def create_nifti_4d(path: Path, shape=(50, 50, 50, 10)):
    """Create a 4D NIfTI file at specified path"""
    data = np.random.rand(*shape)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.to_filename(str(path))
    return path

def create_nifti_5d(path: Path, shape=(50, 50, 50, 3, 10)):
    """Create a 5D NIfTI file at specified path"""
    data = np.random.rand(*shape)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.to_filename(str(path))
    return path

@pytest.fixture
def sample_3d_nifti(tmp_path):
    return create_nifti_3d(tmp_path / "test3d.nii.gz")

@pytest.fixture
def sample_4d_nifti(tmp_path):
    return create_nifti_4d(tmp_path / "test4d.nii.gz")

@pytest.fixture
def sample_5d_nifti(tmp_path):
    return create_nifti_5d(tmp_path / "test5d.nii.gz") 

@pytest.fixture
def sample_mask(tmp_path):
    return create_nifti_3d(tmp_path / "mask.nii.gz", mask=True)

@pytest.fixture
def sample_underlay(tmp_path):
    return create_nifti_3d(tmp_path / "underlay.nii.gz")

@pytest.fixture
def example_fixture():
    return "example"

def create_bids_hierarchy(base_path: Path, structure: dict):
    """
    Creates a BIDS directory structure from a nested dictionary specification.
    
    Args:
        base_path: Path to create the structure under
        structure: Dictionary format: {
            "sub-01": {
                "ses-A": {
                    "anat": ["T1w", "FLAIR"],
                    "func": ["bold"]
                },
                "ses-B": {...}
            },
            ...
        }
    """
    for subject, sessions in structure.items():
        for session, modalities in sessions.items():
            for modality, scans in modalities.items():
                mod_path = base_path / subject / session / modality
                mod_path.mkdir(parents=True, exist_ok=True)
                
                for scan in scans:
                    scan_name = f"{subject}_{session}_desc-{scan}.nii.gz"
                    create_nifti_3d(mod_path / scan_name)

@pytest.fixture
def sample_bids_structure(tmp_path):
    """Create a sample BIDS structure for testing"""
    structure = {
        "sub-01": {
            "ses-01": {
                "anat": ["T1w", "FLAIR"],
                "func": ["bold"]
            }
        },
        "sub-02": {
            "ses-02": {
                "dwi": ["dti"]
            }
        }
    }
    bids_dir = tmp_path / "bids"
    create_bids_hierarchy(bids_dir, structure)
    
    return bids_dir

@pytest.fixture
def empty_bids_dir(tmp_path):
    """Create an empty BIDS directory"""
    bids_dir = tmp_path / "empty_bids"
    bids_dir.mkdir()
    return bids_dir

