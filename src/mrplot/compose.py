"""Tile rendered PNGs into one labelled figure."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mrplot.media import write_png
from mrplot.rendering import RenderConfig, load_font, wrap_title

RGB = tuple[int, int, int]

# Label text is this fraction of a tile's shorter side, so it reads the same at any resolution.
_FONT_FRACTION = 1 / 30
_MIN_FONT_SIZE = 12
_TITLE_SCALE = 1.25


@dataclass(frozen=True, slots=True)
class Panel:
    """One image in a composed figure, with an optional label drawn under it."""

    path: str | os.PathLike[str]
    label: str | None = None


def panels_from_directories(
    directories: Sequence[tuple[str | None, str | os.PathLike[str]]], rows: Sequence[str],
) -> list[Panel]:
    """Build a grid with one row per pattern and one column per render output folder.

    Each cell is the single PNG in that folder whose name ends in ``_<row>.png``,
    e.g. ``rows=("axial", "coronal")`` picks ``*_axial.png`` then ``*_coronal.png``.
    """
    if not rows:
        raise ValueError("At least one row pattern is required.")
    panels = []
    for row in rows:
        for label, directory in directories:
            folder = Path(directory)
            if not folder.is_dir():
                raise ValueError(f"Not a directory: {folder}")
            matches = sorted(folder.glob(f"*_{row}.png"))
            if len(matches) != 1:
                found = ", ".join(path.name for path in matches) or "none"
                raise ValueError(f"Expected one *_{row}.png in {folder}, found {found}.")
            panels.append(Panel(matches[0], label))
    return panels


def _open_rgb(path: str | os.PathLike[str], background: RGB) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, background)
            canvas.paste(rgba, mask=rgba)
            return canvas
        return image.convert("RGB")


def _line_height(draw: ImageDraw.ImageDraw, font: object) -> int:
    box = draw.textbbox((0, 0), "Ag", font=font)
    return int(box[3] - box[1])


def _draw_lines(
    draw: ImageDraw.ImageDraw, lines: Sequence[str], *, left: int, top: int, width: int,
    font: object, color: RGB, padding: int,
) -> None:
    height = _line_height(draw, font)
    reference = draw.textbbox((0, 0), "Ag", font=font)
    for index, line in enumerate(lines):
        line_width = draw.textlength(line, font=font)
        x = left + max((width - line_width) / 2, 0)
        y = top + index * (height + padding) - reference[1]
        draw.text((x, y), line, font=font, fill=color)


def compose_images(
    panels: Sequence[Panel],
    *,
    columns: int | None = None,
    title: str | None = None,
    spacing: tuple[int, int] = (16, 16),
    font_path: str | None = None,
    font_size: int | None = None,
    background_color: RGB = RenderConfig().background_color,
    text_color: RGB = RenderConfig().text_color,
) -> Image.Image:
    """Tile images row by row into one figure and return it.

    Every tile is as large as the largest image, and smaller images are centred
    in their tile. Labels go under each image, and the title above the grid.
    """
    if not panels:
        raise ValueError("At least one image is required.")
    columns = len(panels) if columns is None else columns
    if columns <= 0:
        raise ValueError("columns must be positive.")
    gap_x, gap_y = spacing
    if gap_x < 0 or gap_y < 0:
        raise ValueError("spacing must be nonnegative.")
    if font_size is not None and font_size <= 0:
        raise ValueError("font_size must be positive.")
    for name, color in (("background_color", background_color), ("text_color", text_color)):
        if len(color) != 3 or any(not 0 <= channel <= 255 for channel in color):
            raise ValueError(f"{name} must be three channels in the uint8 range.")

    images = [_open_rgb(panel.path, background_color) for panel in panels]
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    label_size = font_size or max(_MIN_FONT_SIZE, round(min(tile_width, tile_height) * _FONT_FRACTION))
    label_font = load_font(font_path, label_size)
    title_font = load_font(font_path, round(label_size * _TITLE_SCALE))
    padding = max(2, label_size // 4)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_lines = [
        wrap_title(measure, panel.label, tile_width, label_font) if panel.label else []
        for panel in panels
    ]
    max_label_lines = max(len(lines) for lines in label_lines)
    label_line_height = _line_height(measure, label_font)
    label_height = (
        padding + max_label_lines * (label_line_height + padding) if max_label_lines else 0
    )

    rows = -(-len(images) // columns)
    used_columns = min(columns, len(images))
    cell_width = tile_width + 2 * gap_x
    cell_height = tile_height + label_height + 2 * gap_y
    width = used_columns * cell_width

    title_lines = wrap_title(measure, title, max(width - 2 * padding, 1), title_font) if title else []
    title_line_height = _line_height(measure, title_font)
    title_height = (
        padding + len(title_lines) * (title_line_height + padding) if title_lines else 0
    )

    canvas = Image.new("RGB", (width, title_height + rows * cell_height), background_color)
    draw = ImageDraw.Draw(canvas)
    if title_lines:
        _draw_lines(
            draw, title_lines, left=0, top=padding, width=width,
            font=title_font, color=text_color, padding=padding,
        )
    for index, (image, lines) in enumerate(zip(images, label_lines)):
        row, column = divmod(index, columns)
        left = column * cell_width + gap_x
        top = title_height + row * cell_height + gap_y
        canvas.paste(
            image, (left + (tile_width - image.width) // 2, top + (tile_height - image.height) // 2),
        )
        if lines:
            _draw_lines(
                draw, lines, left=left, top=top + tile_height + padding, width=tile_width,
                font=label_font, color=text_color, padding=padding,
            )
    return canvas


def compose_files(
    panels: Sequence[Panel | str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    columns: int | None = None,
    title: str | None = None,
    spacing: tuple[int, int] = (16, 16),
    font_path: str | None = None,
    font_size: int | None = None,
    background_color: RGB = RenderConfig().background_color,
    text_color: RGB = RenderConfig().text_color,
) -> str:
    """Tile PNGs into one figure and write it to ``output``, returning its path.

    Pass plain paths for unlabelled images, or :class:`Panel` objects for labels.
    """
    figure = compose_images(
        [panel if isinstance(panel, Panel) else Panel(panel) for panel in panels],
        columns=columns, title=title, spacing=spacing, font_path=font_path,
        font_size=font_size, background_color=background_color, text_color=text_color,
    )
    return write_png(os.fspath(output), np.asarray(figure))
