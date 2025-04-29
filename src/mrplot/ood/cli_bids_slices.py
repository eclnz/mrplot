import click
import pickle
import logging
from mrplot.ood.bids import BIDS
from mrplot.ood.clogging import logger

@click.command()
@click.argument('bids_dir', type=click.Path(exists=True))
@click.argument('scan_regex', type=str)
@click.argument('origin_regex', type=str)
@click.option('--required-dims', '-rd', type=int, default=None, help='Required number of dimensions for scans (optional)')
@click.option('--slice-indices', '-i', nargs=3, type=int, default=None, help='Three indices for slicing (optional)')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output pickle file for the slices')
@click.option('--log-level', type=click.Choice(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], case_sensitive=False), default='WARNING', help='Set the logging level (default: WARNING)')
def main(bids_dir, scan_regex, origin_regex, required_dims, slice_indices, output, log_level):
    """Create slice objects from a BIDS directory and save them as a pickle file."""
    # Set logging level
    logger.setLevel(getattr(logging, log_level.upper()))

    # Convert slice_indices to tuple if provided
    if slice_indices is not None:
        slice_indices = tuple(slice_indices)
    
    bids = BIDS(bids_dir)
    slices = bids.create_slices(
        scan_regex=scan_regex,
        origin_regex=origin_regex,
        required_dims=required_dims,
        slice_indices=slice_indices
    )
    with open(output, 'wb') as f:
        pickle.dump(slices, f)
    click.echo(f"Saved {len(slices)} slices to {output}")

if __name__ == '__main__':
    main() 