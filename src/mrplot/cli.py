"""
Command-line interface for generating MRI slice plots and videos.
"""

import argparse
import os
from .plotUtils import MRIDataProcessor, MRIPlotter
from .plotConfig import PlotConfig
import importlib.resources as pkg_resources
import click
from .groupPlotter import GroupPlotter, GroupPlotConfig
import json
from pathlib import Path
from importlib import resources
from . import templates

@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    """mrplot - MRI plotting and visualization tool"""
    pass

@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--padding', type=int, default=10, help='Padding around images in pixels (default: 10)')
@click.option('--fps', type=int, default=10, help='Frames per second for videos (default: 10)')
@click.option('--crop', is_flag=True, help='Crop images using mask boundaries')
@click.option('--mask', type=click.Path(exists=True), help='Path to mask NIfTI file for cropping')
@click.option('--underlay', type=click.Path(exists=True), help='Path to underlay NIfTI image')
@click.option('--mask-underlay', is_flag=True, help='Apply mask to underlay image when using --crop')
def main(input, output_dir, padding, fps, crop, mask, underlay, mask_underlay):
    """Generate MRI slice plots/videos from NIfTI files."""
    # Validate inputs
    if not os.path.isfile(input):
        raise click.FileError(input, hint='Input file not found')

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
@click.argument('bids-dir', type=click.Path(exists=True))
@click.option('-o', '--output-dir', required=True, help='Output directory')
@click.option('-s', '--scans', multiple=True, help='Scans to process (e.g. T1w bold)')
@click.option('--config-file', type=click.Path(exists=True), help='JSON config file with plot settings')
@click.option('--interactive', is_flag=True, help='Configure settings interactively')
@click.option('--save-config', type=click.Path(), help='Save configuration to file for reuse')
def group(bids_dir, output_dir, scans, config_file, interactive, save_config):
    """Process group of scans from BIDS directory"""
    # Load or initialize configuration
    if config_file:
        with open(config_file) as f:
            config_data = json.load(f)
        config = GroupPlotConfig(**config_data)
    else:
        config = GroupPlotConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            selected_scans=list(scans),
            all_scans=[]  # Will be populated automatically
        )
    
    # Initialize plotter
    plotter = GroupPlotter(config, None)  # Auto-discover subjects
    
    if interactive:
        configure_interactively(plotter)
    
    # Validate before running
    validate_configuration(plotter.config)
    
    # Save config if requested
    if save_config:
        save_configuration(plotter.config, save_config)
    
    # Run the plotter
    plotter.plot()

def configure_interactively(plotter):
    """Interactive configuration wizard"""
    click.echo("\n=== Interactive Configuration ===")
    
    # Scan selection
    click.echo("\nAvailable scans:")
    for idx, scan in enumerate(plotter.config.all_scans, 1):
        click.echo(f"{idx}: {scan}")
    
    selected = click.prompt(
        "\nSelect scans to process (comma-separated numbers)",
        type=click.IntRange(1, len(plotter.config.all_scans)),
        multiple=True
    )
    plotter.config.selected_scans = [plotter.config.all_scans[i-1] for i in selected]
    
    # Configure each selected scan
    for scan in plotter.config.selected_scans:
        click.echo(f"\nConfiguring {scan}:")
        plotter.scan_configs[scan] = configure_scan_interactively()

def configure_scan_interactively():
    """Configure settings for a single scan"""
    cfg = PlotConfig()
    
    cfg.padding = click.prompt(
        "Padding around images (mm)", 
        type=int, 
        default=cfg.padding
    )
    
    cfg.fps = click.prompt(
        "Animation frame rate (for 4D data)",
        type=int,
        default=cfg.fps
    )
    
    cfg.crop = click.confirm(
        "Automatically crop empty space?",
        default=cfg.crop
    )
    
    # Add more parameters as needed...
    
    return cfg

def validate_configuration(config):
    """Validate configuration before execution"""
    if not config.selected_scans:
        raise click.BadParameter("No scans selected for processing")
    
    if not Path(config.bids_dir).exists():
        raise click.BadParameter(f"BIDS directory not found: {config.bids_dir}")
    
    # Add more validation as needed...

def save_configuration(config, path):
    """Save configuration to JSON file"""
    config_dict = {
        'bids_dir': config.bids_dir,
        'output_dir': config.output_dir,
        'selected_scans': config.selected_scans,
        'scan_configs': {
            scan: vars(cfg) for scan, cfg in config.scan_configs.items()
        }
    }
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)

if __name__ == "__main__":
    cli() 