import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from mrplot.plotConfig import PlotConfig
from mrplot.groupPlotter import GroupPlotter, GroupPlotConfig
from mrplot.plotUtils import MRIDataProcessor
from pathlib import Path
import numpy as np
import nibabel as nib
from tests.conftest import create_nifti_3d, create_nifti_4d, create_nifti_5d, create_bids_hierarchy

# Define the BIDS structure as a constant
BIDS_STRUCTURE = {
    "sub-01": {
        "ses-1": {
            "anat": ["T1w"],
            "func": ["bold", "motion"]
        }
    }
}

@pytest.fixture(scope="module")
def mock_bids_dir(tmp_path_factory):
    """Fixture to create a BIDS directory structure for testing"""
    bids_dir = tmp_path_factory.mktemp("bids")
    create_bids_hierarchy(bids_dir, BIDS_STRUCTURE)
    return bids_dir

@pytest.fixture
def sample_subject_session(mock_bids_dir):
    return {
        'sub-01': {
            'ses-1': {
                'T1w': {
                    'scan_path': str(mock_bids_dir / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz")
                },
                'aMRI': {
                    'scan_path': str(mock_bids_dir / "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz")
                },
                'motion': {
                    'scan_path': str(mock_bids_dir / "sub-01/ses-1/func/sub-01_ses-1_motion.nii.gz")
                }
            }
        }
    }

@pytest.fixture
def mock_config(tmp_path, mock_bids_dir):
    return GroupPlotConfig(
        bids_dir=str(mock_bids_dir),
        selected_scans=['T1w', 'aMRI', 'motion'],
        all_scans=['T1w', 'aMRI', 'motion'],
        output_dir=str(tmp_path / "output")
    )

def test_group_plotter_initialization(mock_config, sample_subject_session):
    plotter = GroupPlotter(mock_config, sample_subject_session)
    
    assert plotter.config == mock_config
    assert plotter.subject_session_list == sample_subject_session
    assert set(plotter.scan_configs.keys()) == {'T1w', 'aMRI', 'motion'}
    assert all(isinstance(cfg, PlotConfig) for cfg in plotter.scan_configs.values())

def test_find_scan_path(sample_subject_session, mock_config):
    plotter = GroupPlotter(mock_config, sample_subject_session)
    
    # Test found path
    expected_path = str(mock_config.bids_dir) + "/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"
    path = plotter._find_scan_path('sub-01', 'ses-1', 'T1w')
    assert path == expected_path
    
    # Test not found
    path = plotter._find_scan_path('sub-01', 'ses-1', 'dwi')
    assert path is None

@patch('builtins.input')
def test_select_scan(mock_input, mock_config, sample_subject_session):
    plotter = GroupPlotter(mock_config, sample_subject_session)
    
    # Test valid selection
    mock_input.return_value = '2'
    selected = plotter._select_scan("Test prompt")
    assert selected == 'aMRI'
    
    # Test invalid then valid selection
    mock_input.side_effect = ['5', '1']
    selected = plotter._select_scan("Test prompt")
    assert selected == 'T1w'
    
    # Test no selection - need to reset side_effect first
    mock_input.side_effect = None
    mock_input.return_value = ''
    selected = plotter._select_scan("Test prompt")
    assert selected is None

def test_skip_non_selected_scans(mock_config, sample_subject_session):
    # Add a non-selected scan to the data
    sample_subject_session['sub-01']['ses-1']['dwi'] = {'scan_path': '/path/to/dwi'}
    
    plotter = GroupPlotter(mock_config, sample_subject_session)
    
    # The non-selected scan should be skipped
    assert 'dwi' not in plotter.scan_configs 
    
def test_plot_with_real_data(tmp_path_factory):
    bids_dir = tmp_path_factory.mktemp("bids")
    output_dir = str(Path("tests") / "outputs" / "group_plotter")
    
    # Test data names
    t1w_path = bids_dir / "T1w.nii.gz"
    bold_path = bids_dir / "bold.nii.gz"
    motion_path = bids_dir / "motion.nii.gz"
    
    # Create test data
    create_nifti_3d(t1w_path)
    create_nifti_4d(bold_path)
    create_nifti_5d(motion_path)
    
    # Create config using real data
    config = GroupPlotConfig(
        bids_dir=str(bids_dir),
        selected_scans=['T1w', 'bold', 'motion'],
        all_scans=['T1w', 'bold', 'motion'],
        output_dir=str(output_dir)
    )
    
    # Create subject session structure
    subject_session = {
        'sub-01': {
            'ses-1': {
                'T1w': {'scan_path': str(t1w_path)},
                'bold': {'scan_path': str(bold_path)},
                'motion': {'scan_path': str(motion_path)}
            }
        }
    }
    
    # Run plotter
    plotter = GroupPlotter(config, subject_session)
    plotter.plot()
    
    # Output paths
    T1w_path = Path(output_dir) / "sub-01/ses-1/T1w_axial.png"
    bold_path = Path(output_dir) / "sub-01/ses-1/bold_sagittal.mp4"
    motion_path = Path(output_dir) / "sub-01/ses-1/motion_coronal.mp4"
    
    # Verify outputs
    assert T1w_path.exists()
    assert bold_path.exists()
    assert motion_path.exists()

if __name__ == "__main__":
    pytest.main([__file__])