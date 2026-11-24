"""Command-line interface for generating MRI slice plots and videos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from nibabel.filebasedimages import ImageFileError

from mrplot.categorical import CATEGORICAL_SCHEMES, CategoricalColors
from mrplot.compose import Panel, compose_files, panels_from_directories
from mrplot.colors import COLOR_SCALES, DEFAULT_CATEGORICAL_SCHEME, HEAT
from mrplot.overlays import ImageOverlay, LabelOverlay
from mrplot.paths import RenderJob, directory_jobs
from mrplot.plotting import render_file
from mrplot.rendering import Interpolation, RenderConfig
from mrplot.slicing import SliceConfig
from mrplot.volumes import validate_volume_path


class _Command(Enum):
    RENDER = "render"
    RENDER_DIRECTORY = "render-directory"
    COMPOSE = "compose"


class _LayerKind(Enum):
    IMAGE = "overlay"
    OUTLINE = "outline"


class _LayerOption(argparse.Action):
    def __call__(
        self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
        values: Any, option_string: str | None = None,
    ) -> None:
        options = getattr(namespace, "layer_options", None)
        if options is None:
            options = []
            namespace.layer_options = options
        kind, _, setting = self.dest.partition("_")
        options.append((_LayerKind(kind), setting, values))


class _PanelOption(argparse.Action):
    """Collect --panel and --images entries in the order they were given."""

    def __call__(
        self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
        values: Any, option_string: str | None = None,
    ) -> None:
        panels = getattr(namespace, self.dest, None) or []
        if option_string == "--panel":
            panels.append((values[0], values[1]))
        else:
            panels.extend((None, value) for value in values)
        setattr(namespace, self.dest, panels)


def _hex_color(value: str) -> tuple[int, int, int]:
    text = value.removeprefix("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError(f"invalid colour {value!r}; expected #RRGGBB")
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid colour {value!r}; expected #RRGGBB") from None


def _rgb_hex(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02X}" for channel in color)


@dataclass(slots=True)
class _LayerSettings:
    opacity: float | None = None
    color_scale: str | None = None
    min: float | None = None
    max: float | None = None
    fill_opacity: float = 0.0
    colors: list[str] | None = None
    colors_file: str | None = None
    scheme: str | None = None
    lut: str | None = None
    background: int = 0


@dataclass
class _LayerArgs:
    kind: _LayerKind
    path: str | None = None
    settings: _LayerSettings = field(default_factory=_LayerSettings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrplot", description="MRI plotting and visualization tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        _Command.RENDER.value,
        help="Render orthogonal (axial/coronal/sagittal) plots or videos for a single NIfTI file.",
    )
    render_parser.add_argument("nifti_file")
    directory_parser = subparsers.add_parser(
        _Command.RENDER_DIRECTORY.value,
        help="Recursively render NIfTI files and mirror their directory structure.",
    )
    directory_parser.add_argument("input_dir")
    for command_parser in (render_parser, directory_parser):
        _add_render_options(command_parser)
    _add_compose_parser(subparsers)
    return parser


def _add_compose_parser(subparsers: Any) -> None:
    defaults = RenderConfig()
    parser = subparsers.add_parser(
        _Command.COMPOSE.value,
        help="Tile rendered PNGs into one labelled figure.",
        description=(
            "Tile PNGs row by row into one figure. Each tile is as large as the largest image, "
            "and smaller images are centred in their tile."
        ),
    )
    parser.add_argument("--output", required=True, help="Path of the PNG to write.")
    parser.add_argument(
        "--panel", nargs=2, metavar=("LABEL", "PNG"), dest="panels", action=_PanelOption,
        help="Add an image with LABEL written under it. Repeat for more panels. "
        "With --rows, PNG is a render output folder instead.",
    )
    parser.add_argument(
        "--images", nargs="+", metavar="PNG", dest="panels", action=_PanelOption,
        help="Add unlabelled images (or folders, with --rows).",
    )
    parser.add_argument(
        "--rows", nargs="+", metavar="PATTERN",
        help="Build the grid from render output folders: one row per pattern (e.g. axial,coronal,sagittal), "
        "one column per folder, each cell the folder's *_<pattern>.png.",
    )
    parser.add_argument(
        "--columns", type=int, default=None,
        help="Panels per row (default: all in one row, or one per folder with --rows).",
    )
    parser.add_argument("--title", default=None, help="Title drawn above the grid.")
    parser.add_argument(
        "--spacing", nargs=2, type=int, metavar=("X", "Y"), default=(16, 16),
        help="Horizontal and vertical gap in pixels around each tile (default: 16 16).",
    )
    parser.add_argument("--font-path", default=None, help="TrueType or OpenType font (default: Pillow's built-in font).")
    parser.add_argument(
        "--font-size", type=int, default=None,
        help="Label size in pixels; the title is 25%% larger (default: scaled to the tile size).",
    )
    parser.add_argument(
        "--background-color", type=_hex_color, default=defaults.background_color, metavar="HEX",
        help=f"Background colour (default: {_rgb_hex(defaults.background_color)}).",
    )
    parser.add_argument(
        "--text-color", type=_hex_color, default=defaults.text_color, metavar="HEX",
        help=f"Text colour (default: {_rgb_hex(defaults.text_color)}).",
    )


def _add_render_options(render_parser: argparse.ArgumentParser) -> None:
    render_parser.add_argument("--mask", dest="mask_path", default=None, help="Mask NIfTI file.")
    render_parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop to the mask's bounding box and zero voxels outside it (requires --mask).",
    )
    render_parser.add_argument(
        "--padding", type=int, default=10, help="Padding (voxels) around the mask when cropping (default: 10)."
    )
    render_parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for 4D data (default: 10)."
    )
    render_parser.add_argument(
        "--frames-as-png",
        action="store_true",
        help="For 4D data, write one PNG per volume (<name>_<plane>_vol-0000.png, ...) instead of an MP4.",
    )
    render_parser.add_argument(
        "--interpolation",
        choices=[interpolation.value for interpolation in Interpolation],
        default=Interpolation.BICUBIC.value,
        help="Resampling method used when scaling a slice up to output size (default: bicubic).",
    )
    render_parser.add_argument(
        "--colorbar", action="store_true", help="Draw a colorbar showing the intensity scale."
    )
    base_colors = render_parser.add_mutually_exclusive_group()
    base_colors.add_argument(
        "--color-scale", choices=tuple(COLOR_SCALES), default="gray",
        help="Color scale applied to the intensity window (default: gray).",
    )
    base_colors.add_argument(
        "--parcellation", choices=CATEGORICAL_SCHEMES,
        help="Render integer labels using this categorical colour scheme.",
    )
    base_colors.add_argument(
        "--parcellation-lut", help="Render integer labels using a hex, FreeSurfer, or FSLeyes lookup table.",
    )
    render_parser.add_argument(
        "--parcellation-background", type=int, help="Background label in a parcellation (default: 0).",
    )
    _add_overlay_options(render_parser)
    render_parser.add_argument(
        "--font-size", type=int, default=RenderConfig().font_size,
        help=f"Title and colorbar label size in pixels (default: {RenderConfig().font_size}).",
    )
    render_parser.add_argument(
        "--font-path", default=None, help="TrueType or OpenType font (default: Pillow's built-in font).",
    )
    titles = render_parser.add_mutually_exclusive_group()
    titles.add_argument(
        "--title", default=None,
        help="Title for each plane. May use {name} (base file name), {layers} (base and drawn layers) "
        "and {plane}; default: '{layers} {plane} Slice'.",
    )
    titles.add_argument("--no-title", action="store_true", help="Draw no title.")
    render_parser.add_argument(
        "--crosshairs",
        action="store_true",
        help="Draw crosshairs on each plane showing where the other two slices were taken from.",
    )
    render_parser.add_argument(
        "--min", dest="intensity_min", type=float, default=None,
        help="Lower intensity bound for the display window (default: minimum of the extracted data).",
    )
    render_parser.add_argument(
        "--max", dest="intensity_max", type=float, default=None,
        help="Upper intensity bound for the display window (default: maximum of the extracted data).",
    )
    render_parser.add_argument("--output-dir", dest="output_dir", required=True)


def _add_overlay_options(parser: argparse.ArgumentParser) -> None:
    images = parser.add_argument_group("image overlay layers", "Repeat --overlay to add another image layer.")
    images.add_argument("--overlay", action=_LayerOption, help="NIfTI image to blend over the base.")
    images.add_argument("--overlay-opacity", type=float, action=_LayerOption, help="Image opacity from 0 to 1 (default: 0.5).")
    images.add_argument(
        "--overlay-color-scale", choices=tuple(COLOR_SCALES), action=_LayerOption,
        help="Colour scale for this overlay (default: heat).",
    )
    images.add_argument("--overlay-min", type=float, action=_LayerOption, help="Lower overlay intensity bound.")
    images.add_argument("--overlay-max", type=float, action=_LayerOption, help="Upper overlay intensity bound.")

    labels = parser.add_argument_group("mask and ROI layers", "Repeat --outline to add another ROI layer.")
    labels.add_argument("--outline", action=_LayerOption, help="NIfTI mask or parcellation to draw over the base.")
    labels.add_argument(
        "--outline-opacity", type=float, action=_LayerOption, help="Line opacity from 0 to 1 (default: 1).",
    )
    labels.add_argument(
        "--outline-fill-opacity", "--mask-opacity", type=float, action=_LayerOption,
        help="Mask or ROI fill opacity from 0 to 1 (default: 0).",
    )
    labels.add_argument(
        "--outline-colors", nargs="+", metavar="HEX", help="One #RRGGBB per foreground label, in ascending order.",
        action=_LayerOption,
    )
    labels.add_argument(
        "--outline-colors-file", action=_LayerOption, help="Text file with one #RRGGBB per foreground label.",
    )
    labels.add_argument(
        "--outline-scheme", choices=CATEGORICAL_SCHEMES, help="Categorical outline colour scheme (default: tableau).",
        action=_LayerOption,
    )
    labels.add_argument(
        "--outline-lut", action=_LayerOption, help="Hex, FreeSurfer, or FSLeyes lookup table for outlines.",
    )
    labels.add_argument(
        "--outline-background", type=int, action=_LayerOption, help="Background label in the outline (default: 0).",
    )


def _validate_render_options(args: argparse.Namespace) -> None:
    if args.crop and args.mask_path is None:
        raise ValueError("--crop requires --mask")
    if args.mask_path is not None:
        validate_volume_path(args.mask_path)
    if args.parcellation_background is not None and args.parcellation is None and args.parcellation_lut is None:
        raise ValueError("--parcellation-background requires --parcellation or --parcellation-lut")


def _jobs_for_command(command: _Command, args: argparse.Namespace) -> Iterator[RenderJob]:
    match command:
        case _Command.RENDER:
            return iter([RenderJob(nifti_file=args.nifti_file, output_dir=args.output_dir)])
        case _Command.RENDER_DIRECTORY:
            return directory_jobs(args.input_dir, args.output_dir)
        case _Command.COMPOSE:
            raise ValueError("compose does not render NIfTI files")


def _parcellation_colors(args: argparse.Namespace) -> CategoricalColors | None:
    background = args.parcellation_background or 0
    if args.parcellation_lut is not None:
        return CategoricalColors.from_lookup_table(args.parcellation_lut, background_label=background)
    if args.parcellation is not None:
        return CategoricalColors(scheme=args.parcellation, background_label=background)
    return None


def _outline_colors(settings: _LayerSettings) -> CategoricalColors:
    sources = (settings.colors, settings.colors_file, settings.lut, settings.scheme)
    if sum(source is not None for source in sources) > 1:
        raise ValueError("Use one colour source per --outline layer.")
    if settings.lut is not None:
        return CategoricalColors.from_lookup_table(settings.lut, background_label=settings.background)
    if settings.colors_file is not None:
        return CategoricalColors.from_color_file(settings.colors_file, background_label=settings.background)
    return CategoricalColors(
        scheme=settings.scheme or DEFAULT_CATEGORICAL_SCHEME,
        background_label=settings.background, colors=settings.colors,
    )


def _make_layer(layer: _LayerArgs) -> ImageOverlay | LabelOverlay:
    path, settings = layer.path, layer.settings
    if path is None:
        flag = layer.kind.value
        raise ValueError(f"{flag.capitalize()} options require --{flag}")
    if layer.kind is _LayerKind.IMAGE:
        return ImageOverlay(
            path, opacity=0.5 if settings.opacity is None else settings.opacity,
            color_scale=HEAT if settings.color_scale is None else COLOR_SCALES[settings.color_scale],
            intensity_min=settings.min, intensity_max=settings.max,
        )
    return LabelOverlay(
        path, colors=_outline_colors(settings), fill_opacity=settings.fill_opacity,
        line_opacity=1 if settings.opacity is None else settings.opacity,
    )


def _overlay_layers(args: argparse.Namespace) -> list[ImageOverlay | LabelOverlay]:
    """Collect per-layer options while preserving the order of image and ROI paths."""
    layers = []
    current: dict[_LayerKind, _LayerArgs] = {}
    for kind, setting, value in getattr(args, "layer_options", ()):
        layer = current.setdefault(kind, _LayerArgs(kind))
        if setting:
            setattr(layer.settings, setting, value)
            continue
        if layer.path is not None:
            layer = current[kind] = _LayerArgs(kind)
        layer.path = value
        layers.append(layer)
    layers.extend(layer for layer in current.values() if layer.path is None)
    return [_make_layer(layer) for layer in layers]


def _run_render_command(args: argparse.Namespace, jobs: Iterable[RenderJob]) -> None:
    overlays = _overlay_layers(args)
    config = RenderConfig(
        interpolation=Interpolation(args.interpolation),
        show_colorbar=args.colorbar,
        font_size=args.font_size,
        color_scale=COLOR_SCALES[args.color_scale],
        slicing=SliceConfig(padding=args.padding, crop=args.crop),
        crosshairs=args.crosshairs,
        fps=args.fps,
        frames_as_png=args.frames_as_png,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        parcellation=_parcellation_colors(args),
        font_path=args.font_path,
        title=args.title,
        show_title=not args.no_title,
    )
    for job in jobs:
        paths = render_file(
            job.nifti_file,
            job.output_dir,
            config=config,
            mask_path=args.mask_path,
            overlays=overlays,
        )
        for path in paths:
            print(path)


def _run_compose_command(args: argparse.Namespace) -> None:
    entries = args.panels or []
    if not entries:
        raise ValueError("compose requires at least one --panel or --images entry")
    columns = args.columns
    if args.rows is not None:
        rows = [row for value in args.rows for row in value.split(",") if row]
        panels = panels_from_directories(entries, rows)
        columns = len(entries) if columns is None else columns
    else:
        panels = [Panel(path, label) for label, path in entries]
    print(compose_files(
        panels, args.output, columns=columns, title=args.title, spacing=tuple(args.spacing),
        font_path=args.font_path, font_size=args.font_size,
        background_color=args.background_color, text_color=args.text_color,
    ))


def cli(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = _Command(args.command)
    try:
        if command is _Command.COMPOSE:
            _run_compose_command(args)
        else:
            _validate_render_options(args)
            _run_render_command(args, _jobs_for_command(command, args))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (OSError, ImageFileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
