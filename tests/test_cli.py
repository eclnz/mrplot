import pytest  # type: ignore
import subprocess
from pathlib import Path
from mrplot.plotUtils import MRIDataProcessor, MRIPlotter
from mrplot.plotConfig import PlotConfig
import os
import imageio  # type: ignore
from click.testing import CliRunner
from mrplot.cli import cli
import json


@pytest.mark.parametrize(
    "data_fixture,expected_media_type",
    [
        ("sample_3d_nifti", "png"),
        ("sample_4d_nifti", "mp4"),
        ("sample_5d_nifti", "mp4"),
    ],
)
def test_cli_dimensionality(data_fixture, expected_media_type, request, tmp_path):
    """Test processing of different dimensionalities (3D/4D/5D)"""
    output_dir = tmp_path / "output"
    data_path = request.getfixturevalue(data_fixture)

    cmd = ["mrplot", "main", str(data_path), str(output_dir)]

    result = subprocess.run(cmd, check=True)

    base_name = os.path.basename(data_path).split(".")[0]
    expected_files = [
        f"{base_name}_axial.{expected_media_type}",
        f"{base_name}_coronal.{expected_media_type}",
        f"{base_name}_sagittal.{expected_media_type}",
    ]

    for fname in expected_files:
        assert (output_dir / fname).exists(), f"Missing {fname}"


def test_cli_with_mask_and_underlay(
    sample_3d_nifti, sample_mask, sample_underlay, tmp_path
):
    """Test mask and underlay integration"""
    output_dir = tmp_path / "output"
    cmd = [
        "mrplot",
        "main",
        str(sample_3d_nifti),
        str(output_dir),
        "--mask",
        str(sample_mask),
        "--underlay",
        str(sample_underlay),
        "--crop",
        "--mask-underlay",
    ]

    result = subprocess.run(cmd, check=True)

    expected_files = ["test3d_axial.png", "test3d_coronal.png", "test3d_sagittal.png"]
    for fname in expected_files:
        assert (output_dir / fname).exists()


def test_cli_invalid_input(tmp_path):
    """Test error handling for invalid input file"""
    output_dir = tmp_path / "output"
    cmd = ["mrplot", "main", "nonexistent.nii.gz", str(output_dir)]

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "Error: Invalid value for 'INPUT'" in exc_info.value.stderr
    assert "Path 'nonexistent.nii.gz' does not exist" in exc_info.value.stderr


def test_cli_output_dir_creation(sample_3d_nifti, tmp_path):
    """Test automatic creation of output directory"""
    output_dir = tmp_path / "new_directory"
    cmd = ["mrplot", "main", str(sample_3d_nifti), str(output_dir)]

    result = subprocess.run(cmd, check=True)
    assert output_dir.exists()


def test_cli_fps_option(sample_4d_nifti, tmp_path):
    """Test FPS option validation and video generation"""
    output_dir = tmp_path / "output"
    cmd = [
        "mrplot",
        "main",
        str(sample_4d_nifti),
        str(output_dir),
        "--fps",
        "10",
    ]

    result = subprocess.run(cmd, check=True)

    # Verify video metadata
    video_path = output_dir / "test4d_axial.mp4"
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    assert meta["fps"] == 10


def test_group_command_auto_discovery(sample_bids_structure, tmp_path):
    """Test simplified group command"""
    runner = CliRunner()
    
    result = runner.invoke(
        cli,
        [
            "group",
            str(sample_bids_structure),
            str(tmp_path / "output"),
            "--scans", "T1w",
            "--scans", "bold"
        ]
    )
    
    assert result.exit_code == 0


def test_group_command_with_config_file(sample_bids_structure, tmp_path):
    """Test basic group command"""
    runner = CliRunner()
    
    result = runner.invoke(
        cli,
        [
            "group",
            str(sample_bids_structure),
            str(tmp_path / "output"),
            "--scans", "T1w"
        ]
    )
    
    assert result.exit_code == 0


def test_group_command_error_handling(empty_bids_dir, tmp_path):
    """Test group command error scenarios"""
    runner = CliRunner()
    
    result = runner.invoke(
        cli,
        [
            "group",
            str(empty_bids_dir),
            str(tmp_path / "output"),
            "--scans", "T1w"
        ]
    )
    assert "No valid BIDS structure" in result.output


if __name__ == "__main__":
    pytest.main()
