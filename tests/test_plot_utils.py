import pytest  # type: ignore
import numpy as np  # type: ignore
import nibabel as nib  # type: ignore
import os
from mrplot.plotUtils import MRIDataProcessor, MRIPlotter
from mrplot.plotConfig import PlotConfig


# Parameterized test function
@pytest.mark.parametrize(
    "data_fixture,expected_media_type",
    [
        ("sample_3d_nifti", "png"),
        ("sample_4d_nifti", "mp4"),
        ("sample_5d_nifti", "mp4"),
    ],
)
@pytest.mark.parametrize("mask_scenario", ["no_mask", "with_mask"])
@pytest.mark.parametrize("underlay_scenario", ["no_underlay", "with_underlay"])
def test_all_combinations(
    data_fixture,
    expected_media_type,
    mask_scenario,
    underlay_scenario,
    request,
    sample_mask,
    sample_underlay,
):
    # Get the actual fixture values
    data_path = request.getfixturevalue(data_fixture)
    mask_path = sample_mask if mask_scenario == "with_mask" else None
    underlay_path = sample_underlay if underlay_scenario == "with_underlay" else None

    # Create output directory with test parameters in path
    test_name = f"{os.path.basename(data_path).split('.')[0]}_{mask_scenario}_{underlay_scenario}"
    output_dir = os.path.abspath(f"tests/outputs/{test_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Configure plot settings based on test scenario
    config = PlotConfig(
        padding=10,
        fps=5,
        crop=(mask_scenario == "with_mask"),
        mask=None,  # Actual mask path is passed separately
        underlay_image=None,  # Actual underlay path is passed separately
        mask_underlay=(
            mask_scenario == "with_mask" and underlay_scenario == "with_underlay"
        ),
    )

    # Process the data
    processor = MRIDataProcessor(
        mri_data_path=data_path,
        config=config,
        underlay_image_path=underlay_path,
        mask_path=mask_path,
    )

    # Verify media type detection
    assert processor.media_type == expected_media_type

    # Generate plots
    plotter = MRIPlotter(
        media_type=processor.media_type,
        mri_data=processor.mri_slices,
        config=config,
        output_dir=output_dir,
        scan_name=test_name,
        underlay_image=processor.underlay_slices,
    )
    plotter.plot()

    # Verify output files
    if expected_media_type == "png":
        expected_files = [
            f"{test_name}_sagittal.png",
            f"{test_name}_coronal.png",
            f"{test_name}_axial.png",
        ]
    else:
        expected_files = [
            f"{test_name}_sagittal.mp4",
            f"{test_name}_coronal.mp4",
            f"{test_name}_axial.mp4",
        ]

    for fname in expected_files:
        path = os.path.join(output_dir, fname)
        assert os.path.exists(path), f"Missing output file: {fname}"
        assert os.path.getsize(path) > 0, f"Empty output file: {fname}"


if __name__ == "__main__":
    pytest.main()
