from dataclasses import replace

import imageio
import nibabel as nib
import numpy as np
import pytest

from mrplot.plotting import plot_slices, render_file
from mrplot.rendering import RenderConfig
from mrplot.slicing import prepare_slices


@pytest.mark.parametrize(
    ("shape", "suffix"),
    [((4, 5, 6), "png"), ((4, 5, 6, 3), "mp4"), ((4, 5, 6, 1), "mp4")],
)
def test_plot_prepared_slices_without_cli(shape, suffix, tmp_path):
    image = nib.Nifti1Image(np.zeros(shape), np.eye(4))
    triplet = prepare_slices(image)

    paths = plot_slices(
        triplet, base_name="scan", output_dir=tmp_path, config=RenderConfig(max_fig_size=32),
    )

    expected = [tmp_path / f"scan_{plane}.{suffix}" for plane in ("sagittal", "coronal", "axial")]
    assert paths == [str(path) for path in expected]
    for path in expected:
        assert path.stat().st_size > 0
        if suffix == "mp4":
            with imageio.get_reader(path) as reader:
                assert reader.count_frames() == shape[-1]


@pytest.mark.parametrize(
    "axial",
    [np.zeros((4, 5, 3)), np.zeros((0, 5))],
)
def test_plot_rejects_invalid_slice_arrays_before_writing(axial, tmp_path):
    triplet = prepare_slices(nib.Nifti1Image(np.zeros((4, 5, 6)), np.eye(4)))
    triplet = replace(triplet, axial=axial)

    with pytest.raises(ValueError, match="slice arrays"):
        plot_slices(triplet, base_name="scan", output_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_plot_rejects_mismatched_timepoints_before_writing(tmp_path):
    triplet = prepare_slices(nib.Nifti1Image(np.zeros((4, 5, 6, 2)), np.eye(4)))
    triplet = replace(triplet, axial=np.zeros((*triplet.axial.shape[:2], 3)))

    with pytest.raises(ValueError, match="same number of timepoints"):
        plot_slices(triplet, base_name="scan", output_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "overrides",
    [{"intensity_min": 2.0}, {"intensity_max": -1.0}],
)
def test_render_file_rejects_invalid_resolved_window(sample_3d_nifti, tmp_path, overrides):
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="Minimum intensity must be less"):
        render_file(sample_3d_nifti, output_dir, config=RenderConfig(**overrides))

    assert not output_dir.exists()
