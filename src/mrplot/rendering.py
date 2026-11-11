"""Prepare layouts and render intensity slices to RGB image frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, NamedTuple, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImagePalette

from mrplot.colors import GRAY, ColorScale
from mrplot.slicing import SliceConfig


class Interpolation(Enum):
    """Resampling method used when scaling a slice up to output size."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


def validate_intensity_window(
    minimum: float | None, maximum: float | None, *, allow_equal: bool = False,
) -> None:
    """Validate the bounds available so far, allowing equality for automatic bounds."""
    if (minimum is not None and not isfinite(minimum)) or (
        maximum is not None and not isfinite(maximum)
    ):
        raise ValueError("Intensity bounds must be finite.")
    if minimum is not None and maximum is not None:
        if minimum > maximum or (minimum == maximum and not allow_equal):
            raise ValueError("Minimum intensity must be less than maximum intensity.")

@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Reusable settings for slice selection, frame rendering, and video output."""

    figure_scale_factor: int = 6
    max_fig_size: int = 1500
    interpolation: Interpolation = Interpolation.BICUBIC
    show_colorbar: bool = False
    font_size: int = 20
    font_path: Optional[str] = None
    pixel_alignment: int = 16
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_padding: int = 4
    background_color: Tuple[int, int, int] = (0, 0, 0)
    crosshair_color: Tuple[int, int, int] = (0, 255, 0)
    crosshair_width: int = 1
    colorbar_width: int = 24
    colorbar_margin: int = 8
    colorbar_min_label_width: int = 70
    colorbar_label_format: str = ".3g"
    color_scale: ColorScale = GRAY
    slicing: SliceConfig = SliceConfig()
    crosshairs: bool = False
    fps: int = 10
    intensity_min: float | None = None
    intensity_max: float | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("Frames per second must be positive.")
        validate_intensity_window(self.intensity_min, self.intensity_max)
        positive_dimensions = (
            ("font_size", self.font_size),
            ("figure_scale_factor", self.figure_scale_factor),
            ("max_fig_size", self.max_fig_size),
            ("pixel_alignment", self.pixel_alignment),
            ("crosshair_width", self.crosshair_width),
            ("colorbar_width", self.colorbar_width),
        )
        for name, value in positive_dimensions:
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        nonnegative_dimensions = (
            ("text_padding", self.text_padding),
            ("colorbar_margin", self.colorbar_margin),
            ("colorbar_min_label_width", self.colorbar_min_label_width),
        )
        for name, value in nonnegative_dimensions:
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        colors = (
            ("text_color", self.text_color),
            ("background_color", self.background_color),
            ("crosshair_color", self.crosshair_color),
        )
        for name, color in colors:
            if any(channel < 0 or channel > np.iinfo(np.uint8).max for channel in color):
                raise ValueError(f"{name} channels must be in the uint8 range.")


_PIL_RESAMPLE = {
    Interpolation.NEAREST: Image.Resampling.NEAREST,
    Interpolation.BILINEAR: Image.Resampling.BILINEAR,
    Interpolation.BICUBIC: Image.Resampling.BICUBIC,
    Interpolation.LANCZOS: Image.Resampling.LANCZOS,
}
_UINT8_MAX = np.iinfo(np.uint8).max


class RenderResources(NamedTuple):
    config: RenderConfig
    resample: Image.Resampling
    font: ImageFont.ImageFont
    palette: ImagePalette.ImagePalette


@dataclass(frozen=True, slots=True)
class FrameLayout:
    source_shape: Tuple[int, int]
    spacing: Tuple[float, float]
    intensity_bounds: Tuple[float, float]
    resources: RenderResources
    output_size: Tuple[int, int] = field(init=False)
    colorbar: Optional[Image.Image] = field(init=False)

    def __post_init__(self) -> None:
        config, _, font, palette = self.resources
        output_size = calculate_output_size(self.source_shape, self.spacing, config)
        colorbar = (
            _prepare_colorbar(output_size, self.intensity_bounds, font, config, palette)
            if config.show_colorbar
            else None
        )
        object.__setattr__(self, "output_size", output_size)
        object.__setattr__(self, "colorbar", colorbar)


def calculate_output_size(
    slice_shape: Tuple[int, int], spacing: Tuple[float, float], config: RenderConfig
) -> Tuple[int, int]:
    """Output (width, height) in pixels, scaled from physical size (voxel
    count x spacing in mm), capped to max_fig_size, and rounded to a macro
    block of config.pixel_alignment for video codec compatibility."""
    slice_height, slice_width = slice_shape
    height_spacing, width_spacing = spacing

    width_px = float(slice_width * width_spacing * config.figure_scale_factor)
    height_px = float(slice_height * height_spacing * config.figure_scale_factor)

    scale_ratio = min(config.max_fig_size / width_px, config.max_fig_size / height_px)
    width_px *= scale_ratio
    height_px *= scale_ratio

    alignment = config.pixel_alignment
    width_px = int((width_px + alignment - 1) // alignment * alignment)
    height_px = int((height_px + alignment - 1) // alignment * alignment)

    return width_px, height_px


def _to_intensity_indices(slice_2d: NDArray[Any], intensity_bounds: Tuple[float, float]) -> NDArray[np.uint8]:
    low, high = intensity_bounds
    span = high - low or 1.0
    clipped = np.clip(slice_2d, low, high)
    return cast(NDArray[np.uint8], ((clipped - low) / span * _UINT8_MAX).astype(np.uint8))


def _apply_color_scale(
    intensities: NDArray[np.uint8], palette: ImagePalette.ImagePalette,
) -> Image.Image:
    image = Image.fromarray(intensities, mode="P")
    image.putpalette(palette)
    return image.convert("RGB")


def _draw_crosshair(
    draw: "ImageDraw.ImageDraw",
    crosshair: Tuple[int, int],
    source_shape: Tuple[int, int],
    output_size: Tuple[int, int],
    config: RenderConfig,
) -> None:
    row, col = crosshair
    source_height, source_width = source_shape
    width_px, height_px = output_size
    x = (col + 0.5) * width_px / source_width
    y = (row + 0.5) * height_px / source_height
    draw.line([(x, 0), (x, height_px)], fill=config.crosshair_color, width=config.crosshair_width)
    draw.line([(0, y), (width_px, y)], fill=config.crosshair_color, width=config.crosshair_width)


def _draw_title(
    draw: "ImageDraw.ImageDraw",
    title: str,
    width_px: int,
    font: ImageFont.ImageFont,
    config: RenderConfig,
) -> None:
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_x = max((width_px - (text_bbox[2] - text_bbox[0])) // 2, 0)
    draw.text(
        (text_x, config.text_padding - text_bbox[1]), title,
        font=font, fill=config.text_color,
    )


def _prepare_colorbar(
    output_size: Tuple[int, int],
    intensity_bounds: Tuple[float, float],
    font: "ImageFont.ImageFont",
    config: RenderConfig,
    palette: ImagePalette.ImagePalette,
) -> "Image.Image":
    bar_width, margin = config.colorbar_width, config.colorbar_margin
    width, height = output_size
    low, high = intensity_bounds
    high_label = format(high, config.colorbar_label_format)
    low_label = format(low, config.colorbar_label_format)
    label_width = max(
        config.colorbar_min_label_width,
        *(font.getbbox(label)[2] for label in (high_label, low_label)),
    )

    content_width = width + margin + bar_width + margin + label_width
    alignment = config.pixel_alignment
    canvas_width = (content_width + alignment - 1) // alignment * alignment
    canvas = Image.new("RGB", (canvas_width - width, height), color=config.background_color)

    gradient = np.linspace(_UINT8_MAX, 0, height, dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    canvas.paste(_apply_color_scale(gradient, palette), (margin, 0))

    draw = ImageDraw.Draw(canvas)
    label_x = margin + bar_width + margin
    high_bbox = draw.textbbox((0, 0), high_label, font=font)
    low_bbox = draw.textbbox((0, 0), low_label, font=font)
    draw.text(
        (label_x, config.text_padding - high_bbox[1]), high_label,
        font=font, fill=config.text_color,
    )
    draw.text(
        (label_x, height - config.text_padding - low_bbox[3]), low_label,
        font=font, fill=config.text_color,
    )

    return canvas


def _append_colorbar(
    image: Image.Image, colorbar: Image.Image, background_color: Tuple[int, int, int],
) -> Image.Image:
    width_px, height_px = image.size
    canvas = Image.new("RGB", (width_px + colorbar.width, height_px), color=background_color)
    canvas.paste(image, (0, 0))
    canvas.paste(colorbar, (width_px, 0))
    return canvas


def load_render_resources(config: RenderConfig) -> RenderResources:
    """Load the drawing resources shared by all planes in a plot."""
    if config.font_path is None:
        font = ImageFont.load_default(size=config.font_size)
    else:
        font = ImageFont.truetype(config.font_path, size=config.font_size)
    palette = ImagePalette.ImagePalette(mode="RGB", palette=config.color_scale.palette)
    return RenderResources(
        config=config,
        resample=_PIL_RESAMPLE[config.interpolation],
        font=font,
        palette=palette,
    )


def render_prepared_frame(
    slice_2d: NDArray[Any],
    layout: FrameLayout,
    title: str,
    crosshair: Optional[Tuple[int, int]] = None,
) -> NDArray[np.uint8]:
    """Render a frame using a prepared layout, returning an independently owned array."""
    if slice_2d.shape != layout.source_shape:
        raise ValueError("Slice shape does not match the prepared frame layout.")
    config, resample, font, palette = layout.resources
    image = _apply_color_scale(
        _to_intensity_indices(slice_2d, layout.intensity_bounds), palette,
    )
    image = image.resize(layout.output_size, resample=resample)
    draw = ImageDraw.Draw(image)
    if crosshair is not None:
        _draw_crosshair(draw, crosshair, layout.source_shape, layout.output_size, config)
    if title:
        _draw_title(draw, title, layout.output_size[0], font, config)
    if layout.colorbar is not None:
        image = _append_colorbar(image, layout.colorbar, config.background_color)
    return np.array(image, dtype=np.uint8)


def render_image_frame(
    slice_2d: NDArray[Any],
    spacing: Tuple[float, float],
    config: RenderConfig,
    intensity_bounds: Tuple[float, float],
    title: str,
    crosshair: Optional[Tuple[int, int]] = None,
) -> NDArray[np.uint8]:
    """Prepare and render one 2D intensity slice to an RGB image array with a title."""
    if slice_2d.ndim != 2:
        raise ValueError("Frame rendering requires a 2D slice.")
    source_shape = (slice_2d.shape[0], slice_2d.shape[1])
    layout = FrameLayout(
        source_shape=source_shape,
        spacing=spacing,
        intensity_bounds=intensity_bounds,
        resources=load_render_resources(config),
    )
    return render_prepared_frame(slice_2d, layout, title, crosshair=crosshair)
