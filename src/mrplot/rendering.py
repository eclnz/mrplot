"""Prepare layouts and render intensity or label slices to RGB image frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, NamedTuple, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImagePalette

from mrplot.categorical import CategoricalColors, LabelColors
from mrplot.colors import GRAY, ColorScale
from mrplot.slicing import CropBox, SliceConfig


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


def resolve_intensity_window(
    data_bounds: tuple[float, float], minimum: float | None, maximum: float | None,
) -> tuple[float, float]:
    bounds = (
        data_bounds[0] if minimum is None else minimum,
        data_bounds[1] if maximum is None else maximum,
    )
    validate_intensity_window(*bounds, allow_equal=minimum is None and maximum is None)
    return bounds


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
    frames_as_png: bool = False
    intensity_min: float | None = None
    intensity_max: float | None = None
    parcellation: CategoricalColors | None = None
    title: str | None = None
    show_title: bool = True

    def __post_init__(self) -> None:
        if self.title is not None:
            try:
                format_title(self.title, name="", layers="", plane="")
            except (KeyError, IndexError, ValueError) as error:
                raise ValueError(
                    f"Invalid title template {self.title!r}; use the fields {{name}}, {{layers}} and {{plane}}."
                ) from error
        if self.fps <= 0:
            raise ValueError("Frames per second must be positive.")
        if self.parcellation is not None and (
            self.show_colorbar or self.intensity_min is not None or self.intensity_max is not None
        ):
            raise ValueError("The intensity window and colorbar require continuous-intensity rendering.")
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


DEFAULT_TITLE = "{layers} {plane} Slice"


def format_title(template: str, *, name: str, layers: str, plane: str) -> str:
    """Fill a title template's ``{name}``, ``{layers}`` and ``{plane}`` fields."""
    return template.format(name=name, layers=layers, plane=plane)


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
    label_colors: LabelColors | None = None
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

    return max(alignment, width_px), max(alignment, height_px)


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


def _text_width(draw: "ImageDraw.ImageDraw", text: str, font: ImageFont.ImageFont) -> float:
    return float(draw.textlength(text, font=font))


def _truncate(draw: "ImageDraw.ImageDraw", text: str, width: float, font: ImageFont.ImageFont) -> str:
    ellipsis = "..."
    while text and _text_width(draw, text + ellipsis, font) > width:
        text = text[:-1]
    return text.rstrip() + ellipsis if text else ellipsis


def wrap_title(
    draw: "ImageDraw.ImageDraw", title: str, width: float, font: ImageFont.ImageFont,
    max_lines: int | None = None,
) -> list[str]:
    """Wrap a title at spaces to fit ``width``, shortening lines that still do not fit.

    With ``max_lines``, the words that would spill past the last line are joined onto
    it, and it is shortened with an ellipsis.
    """
    lines: list[str] = []
    for word in title.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and (
            _text_width(draw, candidate, font) <= width
            or (max_lines is not None and len(lines) >= max_lines)
        ):
            lines[-1] = candidate
        else:
            lines.append(word)
    return [
        line if _text_width(draw, line, font) <= width else _truncate(draw, line, width, font)
        for line in lines
    ]


def _draw_title(
    draw: "ImageDraw.ImageDraw",
    title: str,
    output_size: Tuple[int, int],
    font: ImageFont.ImageFont,
    config: RenderConfig,
) -> None:
    width_px, height_px = output_size
    available = max(width_px - 2 * config.text_padding, 1)
    reference = draw.textbbox((0, 0), "Ag", font=font)
    line_height = reference[3] - reference[1] + config.text_padding
    # Wrap within the top quarter of the plot so the title never covers the slice centre.
    max_lines = max(1, height_px // 4 // line_height)
    for index, line in enumerate(wrap_title(draw, title, available, font, max_lines)):
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_x = max((width_px - (text_bbox[2] - text_bbox[0])) // 2, 0)
        y = config.text_padding + index * line_height - reference[1]
        draw.text((text_x, y), line, font=font, fill=config.text_color)


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


def load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    """Load Pillow's built-in scalable font, or a TrueType/OpenType file."""
    if font_path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(font_path, size=size)


def load_render_resources(config: RenderConfig) -> RenderResources:
    """Load the drawing resources shared by all planes in a plot."""
    font = load_font(config.font_path, config.font_size)
    palette = ImagePalette.ImagePalette(mode="RGB", palette=config.color_scale.palette)
    return RenderResources(
        config=config,
        resample=_PIL_RESAMPLE[config.interpolation],
        font=font,
        palette=palette,
    )


def _sample_label_axis(start: int, stop: int, output_length: int) -> NDArray[np.intp]:
    indices = Image.fromarray(np.arange(start, stop, dtype=np.int32)[None, :])
    resized = indices.resize((output_length, 1), resample=Image.Resampling.NEAREST)
    # Offset for the extra background pixel outside the image itself.
    return np.concatenate(([start - 1], np.asarray(resized).ravel(), [stop])).astype(np.intp) + 1


def _label_neighbourhood(
    indices: NDArray[np.int32], output_size: tuple[int, int], crop: CropBox | None,
) -> NDArray[np.int32]:
    """Sample the viewport and keep one adjacent label at each edge."""
    height, width = indices.shape
    left, top, right, bottom = crop if crop is not None else (0, 0, width, height)
    columns = _sample_label_axis(left, right, output_size[0])
    rows = _sample_label_axis(top, bottom, output_size[1])
    return np.pad(indices, 1)[np.ix_(rows, columns)]


def _region_edges(neighbours: NDArray[np.int32]) -> NDArray[np.bool_]:
    """Find the inner pixel of each four-connected border in the viewport."""
    pixels = neighbours[1:-1, 1:-1]
    edge = (pixels != 0) & (
        (pixels != neighbours[:-2, 1:-1]) | (pixels != neighbours[2:, 1:-1])
        | (pixels != neighbours[1:-1, :-2]) | (pixels != neighbours[1:-1, 2:])
    )
    return np.asarray(edge, dtype=np.bool_)


def _with_alpha(image: Image.Image, alpha: NDArray[np.uint8]) -> Image.Image:
    image.putalpha(Image.fromarray(alpha))
    return image


def prepare_label_layer(
    labels: NDArray[Any], output_size: tuple[int, int], *, colors: LabelColors,
    fill_opacity: float, line_opacity: float, crop: CropBox | None = None,
) -> Image.Image:
    neighbours = _label_neighbourhood(colors.encode(labels), output_size, crop)
    pixels = neighbours[1:-1, 1:-1]
    alpha = np.where(pixels != 0, np.uint8(round(fill_opacity * 255)), np.uint8(0))
    if line_opacity:
        # The fill and the line use the same colour. Combine their alpha at the border.
        edge_opacity = fill_opacity + line_opacity * (1 - fill_opacity)
        alpha[_region_edges(neighbours)] = round(edge_opacity * 255)
    return _with_alpha(Image.fromarray(colors.palette((0, 0, 0))[pixels]), alpha)


def prepare_image_layer(
    data: NDArray[Any], output_size: tuple[int, int], *, color_scale: ColorScale,
    bounds: tuple[float, float], opacity: float, interpolation: Interpolation,
) -> Image.Image:
    visible = np.isfinite(data) & (data != 0)
    low, high = bounds
    values = np.where(visible, data, low)
    indices = (
        np.full(data.shape, 255, dtype=np.uint8) if low == high else _to_intensity_indices(values, bounds)
    )
    palette = ImagePalette.ImagePalette(mode="RGB", palette=color_scale.palette)
    rgb = _apply_color_scale(indices, palette)
    alpha = np.where(visible, np.uint8(round(opacity * 255)), np.uint8(0))
    return _with_alpha(rgb, alpha).resize(output_size, resample=_PIL_RESAMPLE[interpolation])


def prepare_outline(
    labels: NDArray[Any], colors: LabelColors, output_size: tuple[int, int],
) -> Image.Image:
    """Prepare the fully opaque outline of a categorical slice."""
    return prepare_label_layer(labels, output_size, colors=colors, fill_opacity=0, line_opacity=1)


def render_prepared_frame(
    slice_2d: NDArray[Any],
    layout: FrameLayout,
    title: str,
    crosshair: Optional[Tuple[int, int]] = None,
    *,
    outline: Image.Image | None = None,
    overlays: Sequence[Image.Image] = (),
) -> NDArray[np.uint8]:
    """Render a frame using a prepared layout, returning an independently owned array."""
    if slice_2d.shape != layout.source_shape:
        raise ValueError("Slice shape does not match the prepared frame layout.")
    config, resample, font, palette = layout.resources
    if layout.label_colors is None:
        image = _apply_color_scale(
            _to_intensity_indices(slice_2d, layout.intensity_bounds), palette,
        )
    else:
        colors = layout.label_colors
        image = Image.fromarray(colors.palette(config.background_color)[colors.encode(slice_2d)])
        resample = Image.Resampling.NEAREST
    image = image.resize(layout.output_size, resample=resample)
    layers = overlays if outline is None else (*overlays, outline)
    for layer in layers:
        image.paste(layer, mask=layer)
    draw = ImageDraw.Draw(image)
    if crosshair is not None:
        _draw_crosshair(draw, crosshair, layout.source_shape, layout.output_size, config)
    if title:
        _draw_title(draw, title, layout.output_size, font, config)
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
    """Render one 2D slice, resolving automatic categorical colours from that slice."""
    if slice_2d.ndim != 2:
        raise ValueError("Frame rendering requires a 2D slice.")
    source_shape = (slice_2d.shape[0], slice_2d.shape[1])
    layout = FrameLayout(
        source_shape=source_shape,
        spacing=spacing,
        intensity_bounds=intensity_bounds,
        resources=load_render_resources(config),
        label_colors=config.parcellation.resolve(slice_2d) if config.parcellation is not None else None,
    )
    return render_prepared_frame(slice_2d, layout, title, crosshair=crosshair)
