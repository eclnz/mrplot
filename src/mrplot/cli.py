"""
Command-line interface for generating MRI slice plots and videos.
"""

import argparse
import os
from .plotUtils import MRIDataProcessor, MRIPlotter
from .plotConfig import PlotConfig
from .indexingUtils import build_series_list
import importlib.resources as pkg_resources
import click
from .groupPlotter import GroupPlotter, GroupPlotConfig
import json
from pathlib import Path
from importlib import resources
from . import templates


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """mrplot - MRI plotting and visualization tool"""
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--padding",
    type=int,
    default=10,
    help="Padding around images in pixels (default: 10)",
)
@click.option(
    "--fps", type=int, default=10, help="Frames per second for videos (default: 10)"
)
@click.option("--crop", is_flag=True, help="Crop images using mask boundaries")
@click.option(
    "--mask", type=click.Path(exists=True), help="Path to mask NIfTI file for cropping"
)
@click.option(
    "--underlay", type=click.Path(exists=True), help="Path to underlay NIfTI image"
)
@click.option(
    "--mask-underlay",
    is_flag=True,
    help="Apply mask to underlay image when using --crop",
)
def main(input, output_dir, padding, fps, crop, mask, underlay, mask_underlay):
    """Generate MRI slice plots/videos from NIfTI files."""
    # Validate inputs
    if not os.path.isfile(input):
        raise click.FileError(input, hint="Input file not found")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create configuration
    config = PlotConfig(
        padding=padding,
        fps=fps,
        crop=crop,
        mask=mask,
        underlay_image=underlay,
        mask_underlay=mask_underlay,
    )

    # Process data
    processor = MRIDataProcessor(
        mri_data_path=input,
        config=config,
        underlay_image_path=underlay,
        mask_path=mask,
    )

    # Generate output name from input file
    base_name = os.path.basename(input).split(".nii")[0]

    # Create and run plotter
    plotter = MRIPlotter(
        media_type=processor.media_type,
        mri_data=processor.mri_slices,
        config=config,
        output_dir=output_dir,
        scan_name=base_name,
        underlay_image=processor.underlay_slices,
    )

    # When loading internal resources
    with resources.path(templates, "default_template.html") as template_path:
        # Use template_path
        plotter.plot()


@cli.command()
@click.argument("bids-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
@click.option("--interactive", is_flag=True, default=False, help="Configure settings interactively")
def group(bids_dir, output_dir, interactive):
    """Process BIDS directory with specified scans"""
    try:
        config = GroupPlotConfig(
            bids_dir=bids_dir,
            output_dir=output_dir
        )
        
        try:
            plotter = GroupPlotter(config)
            # Set all scans as selected by default
            if not interactive:
                plotter.selected_scans = plotter.all_scans
                # Initialize default configs for all scans
                for scan in plotter.selected_scans:
                    plotter.scan_configs[scan] = PlotConfig()
        except ValueError as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
            
        if interactive:
            configure_interactively(plotter)
            
        plotter.plot()
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

def configure_interactively(plotter):
    """Interactive configuration wizard"""
    click.echo("\n=== Interactive Configuration ===")
    
    # Show available scans
    click.echo("\nAvailable scans:")
    for idx, scan in enumerate(plotter.all_scans, 1):
        click.echo(f"{idx}: {scan}")
    
    # Get scan selection
    selected = click.prompt(
        "\nSelect scans to process (comma-separated numbers)",
        type=click.STRING
    )
    
    try:
        selected_indices = [int(i.strip()) for i in selected.split(",")]
        plotter.selected_scans = [plotter.all_scans[i-1] for i in selected_indices]
    except (ValueError, IndexError):
        click.echo("Invalid selection. Please enter comma-separated numbers.")
        raise click.Abort()
    
    # Configure each selected scan
    for scan in plotter.selected_scans:
        click.echo(f"\nConfiguring {scan}:")
        config = PlotConfig()
        
        # Get mask
        if click.confirm("Add a mask?", default=False):
            config.mask = click.prompt("Mask scan name", type=str)
            
        # Get underlay
        if click.confirm("Add an underlay?", default=False):
            config.underlay_image = click.prompt("Underlay scan name", type=str)
            
        # Other options
        config.crop = click.confirm("Crop empty space?", default=False)
        config.padding = click.prompt("Padding (pixels)", type=int, default=10)
        config.fps = click.prompt("FPS (for 4D data)", type=int, default=10)
        
        plotter.scan_configs[scan] = config


if __name__ == "__main__":
    cli()
 