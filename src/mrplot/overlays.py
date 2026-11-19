"""Specifications and preparation for ordered NIfTI overlay layers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from mrplot.categorical import CategoricalColors
from mrplot.colors import HEAT, ColorScale
from mrplot.rendering import (
    Interpolation,
    prepare_image_layer,
    prepare_label_layer,
    resolve_intensity_window,
    validate_intensity_window,
)
from mrplot.slicing import Plane, SliceSpec, SliceTriplet, extract_slice_triplet
from mrplot.volumes import align_overlay_volume, load_volume

_MAX_OUTLINE_VALUES = 500


def _validate_opacity(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} opacity must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class ImageOverlay:
    """Continuous overlay with its own intensity window and colour scale."""

    path: str | os.PathLike[str]
    opacity: float = 0.5
    color_scale: ColorScale = HEAT
    intensity_min: float | None = None
    intensity_max: float | None = None

    def __post_init__(self) -> None:
        _validate_opacity(self.opacity, "Image overlay")
        validate_intensity_window(self.intensity_min, self.intensity_max)


@dataclass(frozen=True, slots=True)
class LabelOverlay:
    """Binary mask or parcellation with independently styled fills and outlines."""

    path: str | os.PathLike[str]
    colors: CategoricalColors = CategoricalColors()
    fill_opacity: float = 0.0
    line_opacity: float = 1.0

    def __post_init__(self) -> None:
        _validate_opacity(self.fill_opacity, "Mask fill")
        _validate_opacity(self.line_opacity, "Outline")


@dataclass(frozen=True, slots=True)
class PreparedOverlay:
    slices: SliceTriplet
    render: Callable[[NDArray[Any], tuple[int, int]], Image.Image]

    def frames_for(self, plane: Plane, size: tuple[int, int]) -> Iterator[Image.Image]:
        data: NDArray[Any] = getattr(self.slices, plane.value)
        if data.ndim == 2:
            return repeat(self.render(data, size))
        return (self.render(data[..., frame], size) for frame in range(data.shape[-1]))


def _finite_bounds(triplet: SliceTriplet) -> tuple[float, float]:
    bounds = []
    for data in (triplet.sagittal, triplet.coronal, triplet.axial):
        finite = data[np.isfinite(data)]
        if finite.size:
            bounds.append((float(finite.min()), float(finite.max())))
    if not bounds:
        return 0.0, 0.0
    return min(low for low, _ in bounds), max(high for _, high in bounds)


def prepare_overlay(
    base: nib.Nifti1Image, spec: SliceSpec, layer: ImageOverlay | LabelOverlay,
    interpolation: Interpolation,
) -> PreparedOverlay:
    """Align and validate one layer before creating any output files."""
    kind = "Outline image" if isinstance(layer, LabelOverlay) else "Image overlay"
    name = f"{kind} ({layer.path})"
    volume = align_overlay_volume(base, load_volume(layer.path), name=name)
    if isinstance(layer, LabelOverlay):
        colors = layer.colors.resolve(
            np.asanyarray(volume.dataobj), name=name, max_values=_MAX_OUTLINE_VALUES,
        )
        slices, _ = extract_slice_triplet(volume, spec)
        return PreparedOverlay(slices, partial(
            prepare_label_layer, colors=colors, fill_opacity=layer.fill_opacity, line_opacity=layer.line_opacity,
        ))

    if volume.get_data_dtype().kind not in "buif":  # type: ignore[no-untyped-call]
        raise ValueError(f"{name} must contain real-valued voxels.")
    slices, _ = extract_slice_triplet(volume, spec)
    bounds = resolve_intensity_window(_finite_bounds(slices), layer.intensity_min, layer.intensity_max)
    return PreparedOverlay(slices, partial(
        prepare_image_layer, color_scale=layer.color_scale, bounds=bounds,
        opacity=layer.opacity, interpolation=interpolation,
    ))
