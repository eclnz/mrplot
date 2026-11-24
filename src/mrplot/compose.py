"""Tile rendered PNGs into one labelled figure."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def _validate_options(
    panels: Sequence[Panel], columns: int, spacing: tuple[int, int],
    font_size: int | None, colors: dict[str, RGB],
) -> None:
    if not panels:
        raise ValueError("At least one image is required.")
    if columns <= 0:
        raise ValueError("columns must be positive.")
    if min(spacing) < 0:
        raise ValueError("spacing must be nonnegative.")
    if font_size is not None and font_size <= 0:
        raise ValueError("font_size must be positive.")
    for name, color in colors.items():
        if len(color) != 3 or any(not 0 <= channel <= 255 for channel in color):
            raise ValueError(f"{name} must be three channels in the uint8 range.")


def default_font_size(tile_size: tuple[int, int]) -> int:
    """Label size that reads the same at any resolution: a fraction of the tile's shorter side."""
    return max(_MIN_FONT_SIZE, round(min(tile_size) * _FONT_FRACTION))


_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@dataclass(frozen=True, slots=True)
class _TextBlock:
    """Centred lines of wrapped text, with ``padding`` above and below each line."""

    lines: Sequence[str]
    font: ImageFont.ImageFont
    padding: int

    @classmethod
    def wrap(cls, text: str | None, width: int, font: ImageFont.ImageFont, padding: int) -> _TextBlock:
        lines = wrap_title(_MEASURE, text, max(width, 1), font) if text else []
        return cls(lines, font, padding)

    @property
    def line_height(self) -> int:
        box = _MEASURE.textbbox((0, 0), "Ag", font=self.font)
        return int(box[3] - box[1])

    def height(self, line_count: int | None = None) -> int:
        """Height of the block, or of a block with ``line_count`` lines in this style."""
        count = len(self.lines) if line_count is None else line_count
        return self.padding + count * (self.line_height + self.padding) if count else 0

    def draw(self, draw: ImageDraw.ImageDraw, *, left: int, top: int, width: int, color: RGB) -> None:
        ascent_offset = draw.textbbox((0, 0), "Ag", font=self.font)[1]
        for index, line in enumerate(self.lines):
            x = left + max((width - draw.textlength(line, font=self.font)) / 2, 0)
            y = top + self.padding + index * (self.line_height + self.padding) - ascent_offset
            draw.text((x, y), line, font=self.font, fill=color)


@dataclass(frozen=True, slots=True)
class _GridLayout:
    """Pixel geometry of a grid of equal cells, each a tile with a label band beneath."""

    count: int
    columns: int
    tile_size: tuple[int, int]
    label_height: int
    spacing: tuple[int, int]
    title_height: int = 0

    @property
    def cell_size(self) -> tuple[int, int]:
        (tile_width, tile_height), (gap_x, gap_y) = self.tile_size, self.spacing
        return tile_width + 2 * gap_x, tile_height + self.label_height + 2 * gap_y

    @property
    def size(self) -> tuple[int, int]:
        cell_width, cell_height = self.cell_size
        rows = -(-self.count // self.columns)
        return min(self.columns, self.count) * cell_width, self.title_height + rows * cell_height

    def tile_origin(self, index: int) -> tuple[int, int]:
        """Top-left corner of the tile area for the panel at ``index``."""
        row, column = divmod(index, self.columns)
        (cell_width, cell_height), (gap_x, gap_y) = self.cell_size, self.spacing
        return column * cell_width + gap_x, self.title_height + row * cell_height + gap_y


def _paste_centred(canvas: Image.Image, image: Image.Image, origin: tuple[int, int], size: tuple[int, int]) -> None:
    (left, top), (width, height) = origin, size
    canvas.paste(image, (left + (width - image.width) // 2, top + (height - image.height) // 2))


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
    columns = len(panels) if columns is None else columns
    _validate_options(
        panels, columns, spacing, font_size,
        {"background_color": background_color, "text_color": text_color},
    )
    images = [_open_rgb(panel.path, background_color) for panel in panels]
    tile_size = (max(image.width for image in images), max(image.height for image in images))

    label_size = font_size or default_font_size(tile_size)
    padding = max(2, label_size // 4)
    label_font = load_font(font_path, label_size)
    labels = [_TextBlock.wrap(panel.label, tile_size[0], label_font, padding) for panel in panels]
    label_height = labels[0].height(max(len(label.lines) for label in labels))

    grid = _GridLayout(len(images), columns, tile_size, label_height, spacing)
    title_font = load_font(font_path, round(label_size * _TITLE_SCALE))
    heading = _TextBlock.wrap(title, grid.size[0] - 2 * padding, title_font, padding)
    grid = replace(grid, title_height=heading.height())

    canvas = Image.new("RGB", grid.size, background_color)
    draw = ImageDraw.Draw(canvas)
    heading.draw(draw, left=0, top=0, width=grid.size[0], color=text_color)
    for index, (image, label) in enumerate(zip(images, labels)):
        left, top = grid.tile_origin(index)
        _paste_centred(canvas, image, (left, top), tile_size)
        label.draw(draw, left=left, top=top + tile_size[1], width=tile_size[0], color=text_color)
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
