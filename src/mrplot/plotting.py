"""Render NIfTI files or prepared orthogonal slices to PNGs and MP4s."""

from __future__ import annotations

import os
from typing import Any, NamedTuple

from numpy.typing import NDArray

from mrplot.media import write_mp4_frames, write_png
from mrplot.paths import nifti_basename
from mrplot.rendering import (
    FrameLayout,
    RenderConfig,
    RenderResources,
    load_render_resources,
    render_prepared_frame,
    validate_intensity_window,
)
from mrplot.slicing import Plane, SliceTriplet, prepare_slices
from mrplot.volumes import load_render_volume, load_volume


class PreparedPlane(NamedTuple):
    plane: Plane
    array: NDArray[Any]
    layout: FrameLayout
    title: str
    crosshair: tuple[int, int] | None


def _validate_plot_inputs(triplet: SliceTriplet, config: RenderConfig) -> tuple[float, float]:
    """Check the input for PNG or MP4 output and return its validated intensity bounds."""
    arrays = (triplet.sagittal, triplet.coronal, triplet.axial)
    ndim = triplet.axial.ndim
    if ndim not in (2, 3) or any(
        array.ndim != ndim or array.size == 0 for array in arrays
    ):
        raise ValueError("Expected three nonempty slice arrays, all 2D or all 3D.")
    if ndim == 3 and len({array.shape[-1] for array in arrays}) != 1:
        raise ValueError("All planes must have the same number of timepoints.")
    data_min, data_max = triplet.intensity_bounds
    minimum, maximum = config.intensity_min, config.intensity_max
    bounds = (
        data_min if minimum is None else minimum,
        data_max if maximum is None else maximum,
    )
    if minimum is None or maximum is None:
        validate_intensity_window(*bounds, allow_equal=minimum is None and maximum is None)
    return bounds


def _prepare_plot(
    triplet: SliceTriplet,
    *,
    base_name: str,
    resources: RenderResources,
    intensity_bounds: tuple[float, float],
) -> tuple[PreparedPlane, ...]:
    """Build the per-plane layouts after validating the plot inputs."""
    planes = (
        (Plane.SAGITTAL, triplet.sagittal, triplet.sagittal_spacing, triplet.sagittal_crosshair),
        (Plane.CORONAL, triplet.coronal, triplet.coronal_spacing, triplet.coronal_crosshair),
        (Plane.AXIAL, triplet.axial, triplet.axial_spacing, triplet.axial_crosshair),
    )
    ndim = triplet.axial.ndim
    return tuple(
        PreparedPlane(
            plane=plane,
            array=array[..., None] if ndim == 2 else array,
            layout=FrameLayout(
                source_shape=(array.shape[0], array.shape[1]),
                spacing=spacing,
                intensity_bounds=intensity_bounds,
                resources=resources,
            ),
            title=f"{base_name} {plane.value.capitalize()} Slice",
            crosshair=crosshair if resources.config.crosshairs else None,
        )
        for plane, array, spacing, crosshair in planes
    )


def plot_slices(
    triplet: SliceTriplet,
    *,
    base_name: str,
    output_dir: str | os.PathLike[str],
    config: RenderConfig = RenderConfig(),
) -> list[str]:
    """Write PNGs for (height, width) slices or MP4s for (height, width, time) slices."""
    bounds = _validate_plot_inputs(triplet, config)
    planes = _prepare_plot(
        triplet, base_name=base_name, resources=load_render_resources(config),
        intensity_bounds=bounds,
    )
    is_time_series = triplet.axial.ndim == 3
    suffix = "mp4" if is_time_series else "png"
    written = []
    for prepared in planes:
        frames = (
            render_prepared_frame(
                prepared.array[..., t], prepared.layout,
                f"{prepared.title}, Timepoint {t}" if is_time_series else prepared.title,
                crosshair=prepared.crosshair,
            )
            for t in range(prepared.array.shape[-1])
        )
        path = os.path.join(output_dir, f"{base_name}_{prepared.plane.value}.{suffix}")
        if is_time_series:
            written.append(
                write_mp4_frames(
                    path, frames, fps=config.fps, macro_block_size=config.pixel_alignment,
                )
            )
        else:
            written.append(write_png(path, next(frames)))
    return written


def render_file(
    nifti_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    config: RenderConfig = RenderConfig(),
    mask_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Prepare a NIfTI scan and write one PNG or MP4 for each plane."""
    image = load_render_volume(nifti_file)
    mask_image = load_volume(mask_path) if mask_path is not None else None
    triplet = prepare_slices(image, mask_image=mask_image, config=config.slicing)
    return plot_slices(
        triplet, base_name=nifti_basename(nifti_file), output_dir=output_dir, config=config,
    )
