import nibabel as nib
import nibabel.arrayproxy as arrayproxy
import numpy as np
import pytest

from mrplot.slicing import (
    AxisBounds,
    CropBounds,
    SliceConfig,
    SliceSpec,
    apply_mask_to_triplet,
    extract_slice_triplet,
    find_crop_bounds,
    reorient_slice_for_display,
    spec_from_mask,
)


def _naive_canonical_triplet(path):
    """Reference: fully reorient, then take the middle slice directly."""
    image = nib.load(path)
    canonical = nib.as_closest_canonical(image)
    data = np.asarray(canonical.dataobj)
    x, y, z = (s // 2 for s in canonical.shape[:3])
    return (
        reorient_slice_for_display(data[:, :, z]),
        reorient_slice_for_display(data[:, y, :]),
        reorient_slice_for_display(data[x, :, :]),
    )


def test_matches_naive_canonical_for_identity_affine(sample_3d_nifti):
    image = nib.load(sample_3d_nifti)
    triplet, _ = extract_slice_triplet(image)
    axial, coronal, sagittal = _naive_canonical_triplet(sample_3d_nifti)
    np.testing.assert_array_equal(triplet.axial, axial)
    np.testing.assert_array_equal(triplet.coronal, coronal)
    np.testing.assert_array_equal(triplet.sagittal, sagittal)


def test_matches_naive_canonical_for_flipped_permuted_affine(sample_3d_nifti_flipped_affine):
    image = nib.load(sample_3d_nifti_flipped_affine)
    triplet, _ = extract_slice_triplet(image)
    axial, coronal, sagittal = _naive_canonical_triplet(sample_3d_nifti_flipped_affine)
    np.testing.assert_array_equal(triplet.axial, axial)
    np.testing.assert_array_equal(triplet.coronal, coronal)
    np.testing.assert_array_equal(triplet.sagittal, sagittal)


def test_4d_keeps_time_axis(sample_4d_nifti):
    image = nib.load(sample_4d_nifti)
    triplet, _ = extract_slice_triplet(image)
    assert triplet.axial.ndim == 3
    assert triplet.axial.shape[-1] == image.shape[-1]


def test_rejects_2d_input():
    image = nib.Nifti1Image(np.zeros((10, 10)), affine=np.eye(4))
    with pytest.raises(ValueError):
        extract_slice_triplet(image)


def test_slice_spec_rejects_negative_indices_at_construction():
    with pytest.raises(ValueError, match="Slice indices must be nonnegative"):
        SliceSpec(indices=(-1, 2, 3))


def test_slice_spec_rejects_indices_outside_its_crop():
    bounds = CropBounds(AxisBounds(2, 4), AxisBounds(1, 3), AxisBounds(0, 5))
    with pytest.raises(ValueError, match="Slice indices must lie within"):
        SliceSpec(indices=(1, 2, 3), crop_bounds=bounds)


@pytest.mark.parametrize(("start", "end"), [(-1, 3), (3, 2)])
def test_axis_bounds_reject_invalid_ranges(start, end):
    with pytest.raises(ValueError, match="Axis bounds must satisfy"):
        AxisBounds(start, end)


def test_never_materializes_the_full_array(sample_4d_nifti, monkeypatch):
    """ArrayProxy.__array__ is what full materialization goes through."""

    def _forbidden_full_load(self, *args, **kwargs):
        raise AssertionError("Full array materialization occurred; slicing is not lazy.")

    monkeypatch.setattr(arrayproxy.ArrayProxy, "__array__", _forbidden_full_load)

    image = nib.load(sample_4d_nifti)
    triplet, _ = extract_slice_triplet(image)
    assert triplet.axial.shape[-1] == image.shape[-1]


def test_find_crop_bounds_matches_mask_extent(sample_mask):
    mask_image = nib.as_closest_canonical(nib.load(sample_mask))
    mask_data = np.asarray(mask_image.dataobj)
    bounds = find_crop_bounds(mask_data)
    nonzero = np.where(np.any(mask_data > 0, axis=(1, 2)))[0]
    assert bounds.sagittal.start == nonzero[0]
    assert bounds.sagittal.end == nonzero[-1]


def test_find_crop_bounds_rejects_empty_mask():
    with pytest.raises(ValueError):
        find_crop_bounds(np.zeros((10, 10, 10)))


def test_slice_config_rejects_negative_padding():
    with pytest.raises(ValueError, match="Padding must be nonnegative"):
        SliceConfig(padding=-1)


def test_spec_from_mask_with_crop_centers_within_bounds(sample_mask):
    mask_image = nib.load(sample_mask)
    spec = spec_from_mask(mask_image, config=SliceConfig(padding=2, crop=True))
    assert spec.crop_bounds is not None
    assert spec.crop_bounds.sagittal.start <= spec.indices[0] <= spec.crop_bounds.sagittal.end


def test_spec_from_mask_without_crop_has_no_bounds(sample_mask):
    mask_image = nib.load(sample_mask)
    spec = spec_from_mask(mask_image, config=SliceConfig(padding=2))
    assert spec.crop_bounds is None


def test_crosshair_marks_the_other_planes_intersection(sample_3d_nifti):
    # The three planes all pass through spec.indices by definition, so a
    # marker placed there must show up at each plane's own crosshair pixel.
    image = nib.load(sample_3d_nifti)
    data = np.asarray(image.dataobj).copy()
    data[10, 20, 30] = 12345.0
    marked_image = nib.Nifti1Image(data, affine=image.affine)

    triplet, _ = extract_slice_triplet(marked_image, SliceSpec(indices=(10, 20, 30)))

    assert triplet.axial[triplet.axial_crosshair] == 12345.0
    assert triplet.coronal[triplet.coronal_crosshair] == 12345.0
    assert triplet.sagittal[triplet.sagittal_crosshair] == 12345.0


def test_apply_mask_to_triplet_zeros_outside_mask(sample_3d_nifti, sample_mask):
    image = nib.load(sample_3d_nifti)
    mask_image = nib.load(sample_mask)
    spec = spec_from_mask(mask_image, config=SliceConfig(padding=0, crop=True))

    triplet, spec = extract_slice_triplet(image, spec)
    mask_triplet, _ = extract_slice_triplet(mask_image, spec)
    masked = apply_mask_to_triplet(triplet, mask_triplet)

    assert np.all(masked.axial[mask_triplet.axial == 0] == 0)
    assert np.any(masked.axial[mask_triplet.axial != 0] != 0)
