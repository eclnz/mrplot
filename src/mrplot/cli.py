"""
Command-line interface for generating MRI slice plots and videos.
"""

import argparse
import os
from .plotUtils import MRIDataProcessor, MRIPlotter
from .plotConfig import PlotConfig
import importlib.resources as pkg_resources
from . import templates  # Create a templates/ directory in your package


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate MRI slice plots/videos from NIfTI files",
        epilog="Example: mriplot input.nii.gz outputs/ --mask mask.nii.gz --fps 15",
    )

    # Required arguments
    parser.add_argument("input", help="Path to input NIfTI file")
    parser.add_argument("output_dir", help="Directory to save output files")

    # Configuration options
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding around images in pixels (default: 10)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for videos (default: 10)"
    )
    parser.add_argument(
        "--crop", action="store_true", help="Crop images using mask boundaries"
    )
    parser.add_argument(
        "--mask", metavar="PATH", help="Path to mask NIfTI file for cropping"
    )
    parser.add_argument(
        "--underlay", metavar="PATH", help="Path to underlay NIfTI image"
    )
    parser.add_argument(
        "--mask-underlay",
        action="store_true",
        help="Apply mask to underlay image when using --crop",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create configuration
    config = PlotConfig(
        padding=args.padding,
        fps=args.fps,
        crop=args.crop,
        mask=args.mask,
        underlay_image=args.underlay,
        mask_underlay=args.mask_underlay,
    )

    # Process data
    processor = MRIDataProcessor(
        mri_data_path=args.input,
        config=config,
        underlay_image_path=args.underlay,
        mask_path=args.mask,
    )

    # Generate output name from input file
    base_name = os.path.basename(args.input).split(".nii")[0]

    # Create and run plotter
    plotter = MRIPlotter(
        media_type=processor.media_type,
        mri_data=processor.mri_slices,
        config=config,
        output_dir=args.output_dir,
        scan_name=base_name,
        underlay_image=processor.underlay_slices,
    )

    # When loading internal resources
    with pkg_resources.path(templates, "default_template.html") as template_path:
        # Use template_path
        plotter.plot()


if __name__ == "__main__":
    main()
