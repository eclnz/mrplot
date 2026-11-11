"""Pure functions for extracting orthogonal slices from a volume.

Orientation is computed from the affine and slices are read directly off
`image.dataobj` in native voxel-axis order, then the resulting small
2D(+trailing) arrays are flipped/transposed into RAS display order - the
full volume is never reoriented or materialized
(`nib.as_closest_canonical`/`as_reoriented` does that internally).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


class Plane(Enum):
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    AXIAL = "axial"


@dataclass(frozen=True, slots=True)
class AxisBounds:
    """Inclusive [start, end] voxel index range along one axis."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Axis bounds must satisfy 0 <= start <= end.")


@dataclass(frozen=True, slots=True)
class CropBounds:
    """Crop bounds for the three spatial axes, named rather than positional."""

    sagittal: AxisBounds
    coronal: AxisBounds
    axial: AxisBounds


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """Canonical RAS voxel indices and optional bounds for extracting a SliceTriplet."""

    indices: Tuple[int, int, int]
    crop_bounds: Optional[CropBounds] = None

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.indices):
            raise ValueError("Slice indices must be nonnegative.")
        if self.crop_bounds is not None:
            bounds = (
                self.crop_bounds.sagittal, self.crop_bounds.coronal, self.crop_bounds.axial,
            )
            if any(
                not bound.start <= index <= bound.end
                for index, bound in zip(self.indices, bounds)
            ):
                raise ValueError("Slice indices must lie within the crop bounds.")


@dataclass(frozen=True, slots=True)
class SliceConfig:
    """Settings for choosing and cropping slices from a mask."""

    padding: int = 10
    crop: bool = False

    def __post_init__(self) -> None:
        if self.padding < 0:
            raise ValueError("Padding must be nonnegative.")


@dataclass(frozen=True, slots=True)
class SliceTriplet:
    """The three orthogonal plane slices extracted from a volume.

    Each plane's `*_spacing` is (height_mm_per_px, width_mm_per_px).
    Each plane's `*_crosshair` is (row, col), in that plane's own pixel
    coordinates, of where the other two slices were taken from.
    """

    axial: NDArray[Any]
    axial_spacing: Tuple[float, float]
    axial_crosshair: Tuple[int, int]
    coronal: NDArray[Any]
    coronal_spacing: Tuple[float, float]
    coronal_crosshair: Tuple[int, int]
    sagittal: NDArray[Any]
    sagittal_spacing: Tuple[float, float]
    sagittal_crosshair: Tuple[int, int]
    intensity_bounds: Tuple[float, float]


# For each plane, the RAS axis it's fixed on, and the two RAS axes (in the
# order we want them to appear in the output) that remain free.
_FIXED_RAS_AXIS = {Plane.SAGITTAL: 0, Plane.CORONAL: 1, Plane.AXIAL: 2}
_FREE_RAS_AXES = {
    Plane.SAGITTAL: (1, 2),  # coronal (A), axial (S)
    Plane.CORONAL: (0, 2),  # sagittal (R), axial (S)
    Plane.AXIAL: (0, 1),  # sagittal (R), coronal (A)
}
_BOUND_FIELD = ("sagittal", "coronal", "axial")


def _native_axis_and_direction(image: nib.Nifti1Image, ras_axis: int) -> Tuple[int, int]:
    """Which voxel axis maps to a given RAS axis (0=R,1=A,2=S), and its
    direction (+1 if increasing with the RAS axis, -1 if flipped).

    `io_orientation`'s rows are indexed by voxel axis (row v = (ras_axis,
    direction) for voxel axis v); this inverts that to look up by RAS axis.
    """
    ornt = nib.io_orientation(image.affine)  # type: ignore[no-untyped-call]
    voxel_axis = int(np.where(ornt[:, 0] == ras_axis)[0][0])
    direction = int(ornt[voxel_axis, 1])
    return voxel_axis, direction


def _ras_axis_size(image: nib.Nifti1Image, ras_axis: int) -> int:
    voxel_axis, _ = _native_axis_and_direction(image, ras_axis)
    return int(image.shape[voxel_axis])


def _to_native_index(ras_index: int, native_size: int, direction: int) -> int:
    return ras_index if direction == 1 else native_size - 1 - ras_index


def _to_native_bounds(bounds: AxisBounds, native_size: int, direction: int) -> AxisBounds:
    if direction == 1:
        return bounds
    return AxisBounds(native_size - 1 - bounds.end, native_size - 1 - bounds.start)


def default_spec(image: nib.Nifti1Image) -> SliceSpec:
    """Middle slice along each axis, no cropping."""
    sagittal, coronal, axial = (_ras_axis_size(image, axis) // 2 for axis in range(3))
    return SliceSpec(indices=(sagittal, coronal, axial))


def find_crop_bounds(
    mask_data: NDArray[Any], *, config: SliceConfig = SliceConfig(padding=0),
) -> CropBounds:
    """Bounding box of the non-zero region of an already-RAS-oriented mask array."""
    if mask_data.ndim < 3:
        raise ValueError(
            f"Mask must have at least 3 dimensions, got {mask_data.ndim}."
        )
    binary = mask_data > 0
    if not np.any(binary):
        raise ValueError("Mask contains no non-zero elements; cannot determine crop bounds.")

    bounds: List[AxisBounds] = []
    for axis in range(3):
        other_axes = tuple(a for a in range(binary.ndim) if a != axis)
        nonzero = np.any(binary, axis=other_axes)
        indices = np.where(nonzero)[0]
        start = max(int(indices[0]) - config.padding, 0)
        end = min(int(indices[-1]) + config.padding, mask_data.shape[axis] - 1)
        bounds.append(AxisBounds(start, end))
    return CropBounds(sagittal=bounds[0], coronal=bounds[1], axial=bounds[2])


def crop_bounds_center(bounds: CropBounds) -> Tuple[int, int, int]:
    return (
        (bounds.sagittal.start + bounds.sagittal.end) // 2,
        (bounds.coronal.start + bounds.coronal.end) // 2,
        (bounds.axial.start + bounds.axial.end) // 2,
    )


def reorient_slice_for_display(slice_array: NDArray[Any]) -> NDArray[Any]:
    """Fixed 90-degree rotation, applied once orientation is RAS-correct."""
    return np.flip(np.rot90(slice_array, k=-1, axes=(0, 1)), axis=0)


def _window_start(crop_bounds: Optional[CropBounds], field: str) -> int:
    return getattr(crop_bounds, field).start if crop_bounds is not None else 0


def _extract_plane(
    image: nib.Nifti1Image, plane: Plane, spec: SliceSpec
) -> Tuple[NDArray[Any], Tuple[float, float], Tuple[int, int]]:
    fixed_ras = _FIXED_RAS_AXIS[plane]
    free_ras_a, free_ras_b = _FREE_RAS_AXES[plane]
    zooms = image.header.get_zooms()  # type: ignore[no-untyped-call]

    fixed_voxel_axis, fixed_direction = _native_axis_and_direction(image, fixed_ras)
    fixed_native_size = image.shape[fixed_voxel_axis]
    fixed_native_index = _to_native_index(
        spec.indices[fixed_ras], fixed_native_size, fixed_direction
    )

    index: List[object] = [slice(None)] * len(image.shape)
    index[fixed_voxel_axis] = fixed_native_index

    free_axes_info = {}
    for free_ras in (free_ras_a, free_ras_b):
        voxel_axis, direction = _native_axis_and_direction(image, free_ras)
        native_size = image.shape[voxel_axis]
        if spec.crop_bounds is not None:
            bounds = getattr(spec.crop_bounds, _BOUND_FIELD[free_ras])
            native_bounds = _to_native_bounds(bounds, native_size, direction)
            index[voxel_axis] = slice(native_bounds.start, native_bounds.end + 1)
        free_axes_info[free_ras] = (voxel_axis, direction)

    # Only the voxels covered by this slicer are ever read off disk.
    raw = np.asarray(image.dataobj[tuple(index)])

    voxel_axis_a, direction_a = free_axes_info[free_ras_a]
    voxel_axis_b, direction_b = free_axes_info[free_ras_b]
    spacing_a, spacing_b = float(zooms[voxel_axis_a]), float(zooms[voxel_axis_b])

    # Position of this plane's crosshair in the (ascending RAS, within-crop)
    # frame raw sits in just before reorient_slice_for_display is applied -
    # see _window_start.
    pos_a = spec.indices[free_ras_a] - _window_start(spec.crop_bounds, _BOUND_FIELD[free_ras_a])
    pos_b = spec.indices[free_ras_b] - _window_start(spec.crop_bounds, _BOUND_FIELD[free_ras_b])

    # numpy keeps remaining axes in their original (ascending voxel-axis)
    # order; reorder to (free_ras_a, free_ras_b) and correct for flips.
    if voxel_axis_a > voxel_axis_b:
        raw = np.swapaxes(raw, 0, 1)
        spacing_a, spacing_b = spacing_b, spacing_a
    if direction_a == -1:
        raw = np.flip(raw, axis=0)
    if direction_b == -1:
        raw = np.flip(raw, axis=1)

    # Run a one-hot marker through the same transform as the real pixel
    # data, rather than re-deriving rot90's index mapping by hand.
    marker = np.zeros(raw.shape[:2], dtype=np.uint8)
    marker[pos_a, pos_b] = 1
    crosshair_row, crosshair_col = np.argwhere(reorient_slice_for_display(marker))[0]
    crosshair = (int(crosshair_row), int(crosshair_col))

    display = reorient_slice_for_display(raw)
    # rot90 swaps the axes, so display spacing is (b, a), not (a, b).
    return display, (spacing_b, spacing_a), crosshair


def extract_slice_triplet(
    image: nib.Nifti1Image, spec: Optional[SliceSpec] = None
) -> Tuple[SliceTriplet, SliceSpec]:
    """Extract the axial/coronal/sagittal planes described by `spec`
    (or the middle of the volume if `spec` is None)."""
    if len(image.shape) < 3:
        raise ValueError("The image must have at least 3 dimensions to extract slices.")

    resolved_spec = spec if spec is not None else default_spec(image)

    sagittal, sagittal_spacing, sagittal_crosshair = _extract_plane(image, Plane.SAGITTAL, resolved_spec)
    coronal, coronal_spacing, coronal_crosshair = _extract_plane(image, Plane.CORONAL, resolved_spec)
    axial, axial_spacing, axial_crosshair = _extract_plane(image, Plane.AXIAL, resolved_spec)

    intensity_bounds = (
        float(min(axial.min(), coronal.min(), sagittal.min())),
        float(max(axial.max(), coronal.max(), sagittal.max())),
    )
    triplet = SliceTriplet(
        axial=axial,
        axial_spacing=axial_spacing,
        axial_crosshair=axial_crosshair,
        coronal=coronal,
        coronal_spacing=coronal_spacing,
        coronal_crosshair=coronal_crosshair,
        sagittal=sagittal,
        sagittal_spacing=sagittal_spacing,
        sagittal_crosshair=sagittal_crosshair,
        intensity_bounds=intensity_bounds,
    )
    return triplet, resolved_spec


def spec_from_mask(mask_image: nib.Nifti1Image, *, config: SliceConfig) -> SliceSpec:
    """Derive slice indices (and, if `crop`, crop bounds) centered on a mask."""
    canonical_mask = nib.as_closest_canonical(mask_image)  # type: ignore[no-untyped-call]
    mask_data = np.asarray(canonical_mask.dataobj)
    bounds = find_crop_bounds(mask_data, config=config)
    indices = crop_bounds_center(bounds)
    return SliceSpec(indices=indices, crop_bounds=bounds if config.crop else None)


def _broadcastable(mask: NDArray[Any], like: NDArray[Any]) -> NDArray[Any]:
    while mask.ndim < like.ndim:
        mask = mask[..., np.newaxis]
    return mask


def apply_mask_to_triplet(triplet: SliceTriplet, mask_triplet: SliceTriplet) -> SliceTriplet:
    """Zero out voxels outside the mask, then recompute intensity bounds."""
    axial = triplet.axial * _broadcastable(mask_triplet.axial > 0, triplet.axial)
    coronal = triplet.coronal * _broadcastable(mask_triplet.coronal > 0, triplet.coronal)
    sagittal = triplet.sagittal * _broadcastable(mask_triplet.sagittal > 0, triplet.sagittal)
    intensity_bounds = (
        float(min(axial.min(), coronal.min(), sagittal.min())),
        float(max(axial.max(), coronal.max(), sagittal.max())),
    )
    return SliceTriplet(
        axial=axial,
        axial_spacing=triplet.axial_spacing,
        axial_crosshair=triplet.axial_crosshair,
        coronal=coronal,
        coronal_spacing=triplet.coronal_spacing,
        coronal_crosshair=triplet.coronal_crosshair,
        sagittal=sagittal,
        sagittal_spacing=triplet.sagittal_spacing,
        sagittal_crosshair=triplet.sagittal_crosshair,
        intensity_bounds=intensity_bounds,
    )


def prepare_slices(
    image: nib.Nifti1Image,
    *,
    mask_image: Optional[nib.Nifti1Image] = None,
    config: SliceConfig = SliceConfig(),
) -> SliceTriplet:
    """Extract centered slices, optionally using a mask for selection and cropping."""
    if mask_image is None:
        if config.crop:
            raise ValueError("Cropping requires a mask.")
        return extract_slice_triplet(image)[0]

    if mask_image.shape[:3] != image.shape[:3]:
        raise ValueError(
            "Mask does not match the size of the input volume in the first three dimensions."
        )
    spec = spec_from_mask(mask_image, config=config)
    triplet, _ = extract_slice_triplet(image, spec)
    if config.crop:
        mask_triplet, _ = extract_slice_triplet(mask_image, spec)
        return apply_mask_to_triplet(triplet, mask_triplet)
    return triplet
