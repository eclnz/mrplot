#!/usr/bin/env python3
"""
Create optimized BIDS datasets with reduced file sizes by keeping only selected slices.

This tool helps create lightweight versions of BIDS datasets for faster transfer and visualization
by keeping only the central slices (or specified offsets) in each anatomical plane.
"""

import argparse
import os
import sys
import logging
import traceback
from mrplot.ood.bids import BIDS
from mrplot.ood.clogging import logger

def main():
    parser = argparse.ArgumentParser(
        description="Create optimized BIDS datasets with reduced file sizes by keeping only selected slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_dir", 
                        help="Path to the input BIDS dataset directory")
    parser.add_argument("output_dir", 
                        help="Path where the optimized dataset will be saved")
    parser.add_argument("--scan-regex", type=str, default=".*", 
                        help="Regular expression to filter scan names")
    parser.add_argument("--origin-regex", type=str, default=".*", 
                        help="Regular expression to filter scan origins")
    parser.add_argument("--sagittal-offset", "-so", type=float, default=0.0, 
                        help="Relative offset in sagittal plane (-0.5 to 0.5, where 0 is center)")
    parser.add_argument("--coronal-offset", "-co", type=float, default=0.0, 
                        help="Relative offset in coronal plane (-0.5 to 0.5, where 0 is center)")
    parser.add_argument("--axial-offset", "-ao", type=float, default=0.0, 
                        help="Relative offset in axial plane (-0.5 to 0.5, where 0 is center)")
    parser.add_argument("--full-volume", action="store_true", 
                        help="Keep the full volume instead of only selected slices")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    try:
        # Check if input directory exists
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist or is not a directory.")
            return 1
            
        print(f"Loading BIDS dataset from: {args.input_dir}")
        bids = BIDS(args.input_dir)
        
        # Check if offsets are within valid range
        for name, offset in [("sagittal", args.sagittal_offset), 
                           ("coronal", args.coronal_offset), 
                           ("axial", args.axial_offset)]:
            if offset < -0.5 or offset > 0.5:
                print(f"Warning: {name} offset {offset} is outside valid range (-0.5 to 0.5). Will be clamped.")
        
        slice_offsets = {
            'sagittal': args.sagittal_offset,
            'coronal': args.coronal_offset,
            'axial': args.axial_offset
        }
        
        print(f"Creating optimized dataset at: {args.output_dir}")
        print(f"Scan filter: {args.scan_regex}")
        print(f"Origin filter: {args.origin_regex}")
        print(f"Using slice offsets: sagittal={args.sagittal_offset}, coronal={args.coronal_offset}, axial={args.axial_offset}")
        
        # Run the optimization
        bids.create_optimized_dataset(
            output_path=args.output_dir,
            scan_regex=args.scan_regex,
            origin_regex=args.origin_regex,
            central_slices_only=not args.full_volume,
            slice_offsets=slice_offsets
        )
        
        print(f"Optimized dataset created successfully at: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 