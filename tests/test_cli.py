import pytest
import subprocess
from pathlib import Path
from mrplot.plotUtils import MRIDataProcessor, MRIPlotter
from mrplot.plotConfig import PlotConfig

def test_cli_basic_3d(sample_3d_nifti, tmp_path):
    """Test basic 3D processing with default settings"""
    output_dir = tmp_path / "output"
    cmd = [
        "python", "-m", "mrplot.cli",
        sample_3d_nifti,
        str(output_dir)
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # Verify output files
    expected_files = [
        "test3d_axial.png",
        "test3d_coronal.png",
        "test3d_sagittal.png"
    ]
    for fname in expected_files:
        assert (output_dir / fname).exists(), f"Missing {fname}"

def test_cli_4d_video(sample_4d_nifti, tmp_path):
    """Test 4D video generation with custom FPS"""
    output_dir = tmp_path / "output"
    cmd = [
        "python", "-m", "mrplot.cli",
        sample_4d_nifti,
        str(output_dir),
        "--fps", "5"
    ]
    
    result = subprocess.run(cmd, check=True)
    
    expected_files = [
        "test4d_axial.mp4",
        "test4d_coronal.mp4",
        "test4d_sagittal.mp4"
    ]
    for fname in expected_files:
        assert (output_dir / fname).exists()

def test_cli_with_mask_and_underlay(sample_3d_nifti, sample_mask, sample_underlay, tmp_path):
    """Test mask and underlay integration"""
    output_dir = tmp_path / "output"
    cmd = [
        "python", "-m", "mrplot.cli",
        sample_3d_nifti,
        str(output_dir),
        "--mask", sample_mask,
        "--underlay", sample_underlay,
        "--crop",
        "--mask-underlay"
    ]
    
    result = subprocess.run(cmd, check=True)
    
    expected_files = [
        "test3d_axial.png",
        "test3d_coronal.png",
        "test3d_sagittal.png"
    ]
    for fname in expected_files:
        assert (output_dir / fname).exists()

def test_cli_invalid_input(tmp_path):
    """Test error handling for invalid input file"""
    output_dir = tmp_path / "output"
    cmd = [
        "python", "-m", "mrplot.cli",
        "nonexistent.nii.gz",
        str(output_dir)
    ]
    
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    assert "Input file not found" in exc_info.value.stderr

def test_cli_output_dir_creation(sample_3d_nifti, tmp_path):
    """Test automatic creation of output directory"""
    output_dir = tmp_path / "new_directory"
    cmd = [
        "python", "-m", "mrplot.cli",
        sample_3d_nifti,
        str(output_dir)
    ]
    
    result = subprocess.run(cmd, check=True)
    assert output_dir.exists() 
    
if __name__ == "__main__":
    pytest.main()