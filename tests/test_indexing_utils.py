import pytest
from pathlib import Path
from mrplot.indexingUtils import (
    list_bids_subjects_sessions_scans,
    build_series_list
)
from tests.conftest import create_bids_hierarchy

BIDS_STRUCTURE = {
    "sub-01": {
        "ses-A": {"anat": ["T1w"]},
        "ses-B": {"func": ["bold"]}
    },
    "sub-02": {
        "ses-B": {"anat": ["FLAIR"]}
    }
}

@pytest.fixture
def bids_directory(tmp_path):
    """Fixture to create a BIDS directory structure for testing"""
    bids_root = tmp_path / "bids_data"
    create_bids_hierarchy(bids_root, BIDS_STRUCTURE)
    return bids_root

def test_basic_structure(bids_directory):
    """Test basic BIDS structure with subjects and sessions"""
    result = list_bids_subjects_sessions_scans(str(bids_directory), ".nii.gz")
    
    assert "sub-01" in result
    assert "ses-A" in result["sub-01"]
    assert "T1w.nii.gz" in result["sub-01"]["ses-A"]
    assert "sub-02" in result
    assert "ses-B" in result["sub-02"]
    assert "FLAIR.nii.gz" in result["sub-02"]["ses-B"]

def test_nested_files(bids_directory):
    """Test files in nested subdirectories under derivatives"""
    # Create nested structure: derivatives/process/sub/ses/scan
    nested_path = bids_directory / "derivatives" / "process" / "sub-01" / "ses-A" / "anat"
    nested_path.mkdir(parents=True, exist_ok=True)
    
    # Create a test file in the nested directory
    test_file = nested_path / "sub-01_ses-A_desc-T1w.nii.gz"
    test_file.touch()

    result = list_bids_subjects_sessions_scans(str(bids_directory), ".nii.gz")
    
    # Verify the nested file is included in the results
    assert "sub-01" in result
    assert "ses-A" in result["sub-01"]
    assert "T1w.nii.gz" in result["sub-01"]["ses-A"]

def test_invalid_directory():
    """Test handling of non-existent directory"""
    with pytest.raises(ValueError) as exc_info:
        list_bids_subjects_sessions_scans("/non/existent/path", ".nii.gz")
    assert "does not exist or is not a directory" in str(exc_info.value)

@pytest.fixture
def sample_data():
    return {
        "sub-01": {
            "ses-A": {"T1w": {}, "FLAIR": {}},
            "ses-B": {"DTI": {}}
        },
        "sub-02": {
            "ses-C": {"T1w": {}, "cohort": {}},
            "unknown": {"PET": {}}
        }
    }

def test_basic_functionality(sample_data):
    """Test extraction of unique scan names"""
    result = build_series_list(sample_data)
    assert result == ["DTI", "FLAIR", "PET", "T1w"]

def test_empty_input():
    """Test handling of empty input"""
    result = build_series_list({})
    assert result == []

def test_ordering(sample_data):
    """Test alphabetical ordering of results"""
    result = build_series_list(sample_data)
    assert result == sorted(result)