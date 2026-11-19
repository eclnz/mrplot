"""Render NIfTI files or prepared orthogonal slices to PNGs and MP4s."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from mrplot.categorical import CategoricalColors, LabelColors
from mrplot.media import write_mp4_frames, write_png
from mrplot.overlays import ImageOverlay, LabelOverlay, PreparedOverlay, prepare_overlay
from mrplot.paths import nifti_basename
from mrplot.rendering import (
    FrameLayout,
    RenderConfig,
    RenderResources,
    load_render_resources,
    render_prepared_frame,
    resolve_intensity_window,
)
from mrplot.slicing import Plane, SliceTriplet, prepare_slices_with_spec
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
    if config.parcellation is not None:
        return triplet.intensity_bounds
    return resolve_intensity_window(triplet.intensity_bounds, config.intensity_min, config.intensity_max)


def _prepare_plot(
    triplet: SliceTriplet,
    *,
    base_name: str,
    resources: RenderResources,
    intensity_bounds: tuple[float, float],
    label_colors: LabelColors | None,
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
                label_colors=label_colors,
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
    return _plot_slices(triplet, base_name=base_name, output_dir=output_dir, config=config)


def _plot_slices(
    triplet: SliceTriplet,
    *,
    base_name: str,
    output_dir: str | os.PathLike[str],
    config: RenderConfig,
    label_colors: LabelColors | None = None,
    overlays: Sequence[PreparedOverlay] = (),
) -> list[str]:
    bounds = _validate_plot_inputs(triplet, config)
    if label_colors is None and config.parcellation is not None:
        label_colors = config.parcellation.resolve(triplet.sagittal, triplet.coronal, triplet.axial)
    planes = _prepare_plot(
        triplet, base_name=base_name, resources=load_render_resources(config),
        intensity_bounds=bounds, label_colors=label_colors,
    )
    is_time_series = triplet.axial.ndim == 3
    suffix = "mp4" if is_time_series else "png"
    written = []
    for prepared in planes:
        layer_frames = [layer.frames_for(prepared.plane, prepared.layout.output_size) for layer in overlays]
        frames = (
            render_prepared_frame(
                prepared.array[..., t], prepared.layout,
                f"{prepared.title}, Timepoint {t}" if is_time_series else prepared.title,
                crosshair=prepared.crosshair,
                overlays=[next(layer) for layer in layer_frames],
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
    outline_path: str | os.PathLike[str] | None = None,
    outline_colors: CategoricalColors | None = None,
    overlays: Sequence[ImageOverlay | LabelOverlay] = (),
) -> list[str]:
    """Write orthogonal views, drawing aligned overlay layers in the given order."""
    if outline_path is None and outline_colors is not None:
        raise ValueError("Outline colours require an outline image.")
    image = load_render_volume(nifti_file)
    mask_image = load_volume(mask_path) if mask_path is not None else None
    parcellation = config.parcellation
    label_colors = (
        parcellation.resolve(np.asanyarray(image.dataobj)) if parcellation is not None else None
    )
    triplet, spec = prepare_slices_with_spec(
        image, mask_image=mask_image, config=config.slicing,
        background_label=parcellation.background_label if parcellation is not None else None,
    )
    layers = list(overlays)
    if outline_path is not None:
        layers.append(LabelOverlay(outline_path, colors=outline_colors or CategoricalColors()))
    prepared_layers = [prepare_overlay(image, spec, layer, config.interpolation) for layer in layers]
    return _plot_slices(
        triplet, base_name=nifti_basename(nifti_file), output_dir=output_dir, config=config,
        label_colors=label_colors, overlays=prepared_layers,
    )
