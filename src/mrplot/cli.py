"""Command-line interface for generating MRI slice plots and videos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import List, Optional

from nibabel.filebasedimages import ImageFileError

from mrplot.colors import COLOR_SCALES
from mrplot.paths import RenderJob, directory_jobs
from mrplot.plotting import render_file
from mrplot.rendering import Interpolation, RenderConfig
from mrplot.slicing import SliceConfig
from mrplot.volumes import validate_volume_path


class _Command(Enum):
    RENDER = "render"
    RENDER_DIRECTORY = "render-directory"


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
    return parser


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
        "--interpolation",
        choices=[interpolation.value for interpolation in Interpolation],
        default=Interpolation.BICUBIC.value,
        help="Resampling method used when scaling a slice up to output size (default: bicubic).",
    )
    render_parser.add_argument(
        "--colorbar", action="store_true", help="Draw a colorbar showing the intensity scale."
    )
    render_parser.add_argument(
        "--color-scale", choices=tuple(COLOR_SCALES), default="gray",
        help="Color scale applied to the intensity window (default: gray).",
    )
    render_parser.add_argument(
        "--font-size", type=int, default=RenderConfig().font_size,
        help="Title and colorbar label size in pixels (default: 20).",
    )
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


def _validate_render_options(args: argparse.Namespace) -> None:
    if args.crop and args.mask_path is None:
        raise ValueError("--crop requires --mask")
    if args.mask_path is not None:
        validate_volume_path(args.mask_path)


def _jobs_for_command(command: _Command, args: argparse.Namespace) -> Iterator[RenderJob]:
    match command:
        case _Command.RENDER:
            return iter([RenderJob(nifti_file=args.nifti_file, output_dir=args.output_dir)])
        case _Command.RENDER_DIRECTORY:
            return directory_jobs(args.input_dir, args.output_dir)


def _run_render_command(args: argparse.Namespace, jobs: Iterable[RenderJob]) -> None:
    config = RenderConfig(
        interpolation=Interpolation(args.interpolation),
        show_colorbar=args.colorbar,
        font_size=args.font_size,
        color_scale=COLOR_SCALES[args.color_scale],
        slicing=SliceConfig(padding=args.padding, crop=args.crop),
        crosshairs=args.crosshairs,
        fps=args.fps,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
    )
    for job in jobs:
        paths = render_file(
            job.nifti_file,
            job.output_dir,
            config=config,
            mask_path=args.mask_path,
        )
        for path in paths:
            print(path)


def cli(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = _Command(args.command)
    try:
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
