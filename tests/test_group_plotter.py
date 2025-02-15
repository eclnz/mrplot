import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from mrplot.plotConfig import PlotConfig
from mrplot.groupPlotter import GroupPlotter, GroupPlotConfig
from mrplot.plotUtils import MRIDataProcessor
from pathlib import Path
import numpy as np
import nibabel as nib
from tests.conftest import create_nifti_3d, create_nifti_4d, create_nifti_5d

@pytest.fixture(scope="module")
def mock_bids_dir(tmp_path_factory):
    bids_dir = tmp_path_factory.mktemp("bids")
    sub_dir = bids_dir / "sub-01" / "ses-1"
    
    # Create anatomical
    anat_dir = sub_dir / "anat"
    anat_dir.mkdir(parents=True)
    create_nifti_3d(anat_dir / "sub-01_ses-1_T1w.nii.gz")
    
    # Create functional
    func_dir = sub_dir / "func"
    func_dir.mkdir(parents=True)
    create_nifti_4d(func_dir / "sub-01_ses-1_bold.nii.gz")
    
    # Create motion (5D)
    create_nifti_5d(func_dir / "sub-01_ses-1_motion.nii.gz")
    
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

@patch('mrplot.plotUtils.MRIDataProcessor')
def test_plot_error_handling(mocker, tmp_path, capsys):
    """Test error handling during plot generation"""
    # Setup BIDS-compliant directory structure
    bids_dir = tmp_path / "bids"
    sub_dir = bids_dir / "sub-01" / "ses-1" / "anat"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Create test files with proper BIDS naming
    good_scan = sub_dir / "sub-01_ses-1_T1w.nii.gz"
    good_scan.touch()
    bad_scan = sub_dir / "sub-01_ses-1_aMRI.nii.gz"  # Proper BIDS naming
    bad_scan.touch()

    # Mock user inputs for interactive mode
    mocker.patch("click.prompt", return_value="10")  # Padding
    mocker.patch("click.confirm", return_value=True)  # Crop

    # Initialize plotter with test config
    config = GroupPlotConfig(
        bids_dir=str(bids_dir),
        output_dir=str(tmp_path / "output"),
        selected_scans=["T1w", "aMRI"],
        all_scans=["T1w", "aMRI"]
    )

    # Force error on aMRI processing with proper path
    def mock_processor_side_effect(mri_data_path, config):
        if "aMRI" in str(mri_data_path):
            raise ValueError("Invalid scan data")
        return MRIDataProcessor(mri_data_path, config)

    mocker.patch(
        "mrplot.groupPlotter.MRIDataProcessor",
        side_effect=mock_processor_side_effect
    )

    plotter = GroupPlotter(config, sample_subject_session)
    plotter.plot()

    # Verify outputs
    captured = capsys.readouterr()
    
    # Check error handling
    assert "Error plotting aMRI: Invalid scan data" in captured.out
    
    # Check successful processing
    assert "Successfully processed T1w" in captured.out
    assert (tmp_path / "output" / "sub-01_T1w_axial.png").exists()
    
    # Verify error scan has no output
    assert not (tmp_path / "output" / "sub-01_aMRI_axial.png").exists()

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