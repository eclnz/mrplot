"""Tile rendered PNGs into one labelled figure."""

from __future__ import annotations

import os
import re
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


def split_labels(text: str) -> list[str]:
    r"""Split a label list at commas; ``\,`` is a literal comma."""
    labels, current, characters = [], "", iter(text)
    for character in characters:
        if character == "\\":
            following = next(characters, "")
            current += "," if following == "," else character + following
        elif character == ",":
            labels.append(current.strip())
            current = ""
        else:
            current += character
    return [*labels, current.strip()]


def _unescape(text: str) -> str:
    return text.replace("\\,", ",")


def parse_frames(text: str) -> list[int]:
    """Parse a frame selection such as ``0,2`` or ``0-2,5`` into volume indices."""
    frames: list[int] = []
    for part in filter(None, (piece.strip() for piece in text.split(","))):
        start, dash, stop = part.partition("-")
        try:
            low = int(start)
            high = int(stop) if dash else low
        except ValueError:
            raise ValueError(f"Invalid frame selection {part!r}; use indices like 0,2 or 0-2.") from None
        if low < 0 or high < low:
            raise ValueError(f"Invalid frame range {part!r}.")
        frames.extend(range(low, high + 1))
    if not frames:
        raise ValueError("The frame selection is empty.")
    return frames


def _row_images(
    folder: Path, row: str, frames: Sequence[int] | None,
) -> list[tuple[int | None, Path]]:
    """Return the folder's single ``*_<row>.png``, or its selected ``*_<row>_vol-NNNN.png`` frames."""
    singles = sorted(folder.glob(f"*_{row}.png"))
    if len(singles) > 1:
        raise ValueError(f"Expected one *_{row}.png in {folder}, found {', '.join(p.name for p in singles)}.")
    pattern = re.compile(rf"(.*)_{re.escape(row)}_vol-(\d+)\.png")
    stems: set[str] = set()
    by_index: dict[int, Path] = {}
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            stems.add(match[1])
            by_index[int(match[2])] = path
    if singles and by_index:
        raise ValueError(
            f"{folder} has both {singles[0].name} and *_{row}_vol-NNNN.png frames; keep one render per folder."
        )
    if singles:
        return [(None, singles[0])]
    if not by_index:
        raise ValueError(f"Expected *_{row}.png or *_{row}_vol-NNNN.png frames in {folder}, found none.")
    if len(stems) > 1:
        raise ValueError(f"Frames from several renders in {folder}: {', '.join(sorted(stems))}.")
    selected = sorted(by_index) if frames is None else list(frames)
    missing = [index for index in selected if index not in by_index]
    if missing:
        raise ValueError(f"{folder} has no {row} frame {', '.join(map(str, missing))}.")
    return [(index, by_index[index]) for index in selected]


def _frame_labels(label: str | None, indices: Sequence[int], folder: Path) -> list[str | None]:
    if label is None:
        return [None] * len(indices)
    labels = split_labels(label)
    if len(labels) == len(indices):
        return list(labels)
    if len(labels) == 1:
        return [f"{labels[0]} {index}" for index in indices]
    raise ValueError(f"{len(labels)} labels for {len(indices)} frames in {folder}.")


def panels_from_directories(
    directories: Sequence[tuple[str | None, str | os.PathLike[str]]],
    rows: Sequence[str],
    *,
    frames: Sequence[int] | None = None,
) -> list[Panel]:
    r"""Build a grid with one row per pattern and one or more columns per render output folder.

    For each row, a folder contributes its single PNG ending in ``_<row>.png``. A folder
    of ``--frames-as-png`` output instead contributes one column per ``_<row>_vol-NNNN.png``
    frame, in frame order (only ``frames``, if given). A frame folder's label may list
    one label per frame, separated by commas (``\,`` for a literal comma); a single label
    is numbered with each frame's volume index.
    """
    if not rows:
        raise ValueError("At least one row pattern is required.")
    folders = [(label, Path(directory)) for label, directory in directories]
    for _, folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")
    widths: dict[int, int] = {}
    panels: list[Panel] = []
    for row in rows:
        for position, (label, folder) in enumerate(folders):
            images = _row_images(folder, row, frames)
            if widths.setdefault(position, len(images)) != len(images):
                raise ValueError(
                    f"{folder} gives {widths[position]} images for {rows[0]} but {len(images)} for {row}."
                )
            labels: list[str | None]
            if images[0][0] is None:
                labels = [None if label is None else _unescape(label)]
            else:
                labels = _frame_labels(label, [index for index, _ in images if index is not None], folder)
            panels.extend(Panel(path, text) for (_, path), text in zip(images, labels))
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
