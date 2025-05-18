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
import nibabel as nib
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Create optimized BIDS datasets or single NIfTI files with reduced file sizes by keeping only selected slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_path", 
                        help="Path to the input BIDS dataset directory or single NIfTI file")
    parser.add_argument("output_dir", 
                        help="Path where the optimized dataset or file will be saved")
    parser.add_argument("--scan-regex", type=str, default=".*", 
                        help="Regular expression to filter scan names (only for BIDS directory)")
    parser.add_argument("--origin-regex", type=str, default=".*", 
                        help="Regular expression to filter scan origins (only for BIDS directory)")
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
        # Check if input is a directory or a single file
        if os.path.isdir(args.input_path):
            print(f"Loading BIDS dataset from: {args.input_path}")
            bids = BIDS(args.input_path)
            
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
        elif os.path.isfile(args.input_path) and args.input_path.endswith(('.nii', '.nii.gz')):
            print(f"Processing single NIfTI file: {args.input_path}")
            
            # Load the NIfTI file
            nifti_img = nib.load(args.input_path)
            data = nifti_img.get_fdata()
            
            # Create a new array with zeros
            optimized_data = np.zeros_like(data)
            
            # Calculate slice indices based on offsets
            x_idx = int((0.5 + args.sagittal_offset) * (data.shape[0] - 1))
            y_idx = int((0.5 + args.coronal_offset) * (data.shape[1] - 1))
            z_idx = int((0.5 + args.axial_offset) * (data.shape[2] - 1))
            
            # Ensure indices are within bounds
            x_idx = max(0, min(data.shape[0] - 1, x_idx))
            y_idx = max(0, min(data.shape[1] - 1, y_idx))
            z_idx = max(0, min(data.shape[2] - 1, z_idx))
            
            print(f"Using slices at: sagittal={x_idx}/{data.shape[0]-1}, coronal={y_idx}/{data.shape[1]-1}, axial={z_idx}/{data.shape[2]-1}")
            
            # Copy only the specified slices
            optimized_data[x_idx, :, :] = data[x_idx, :, :]  # Sagittal slice
            optimized_data[:, y_idx, :] = data[:, y_idx, :]  # Coronal slice
            optimized_data[:, :, z_idx] = data[:, :, z_idx]  # Axial slice
            
            # Create the optimized NIfTI image with the same header
            optimized_img = nib.Nifti1Image(
                optimized_data, 
                nifti_img.affine, 
                nifti_img.header
            )
            
            # Determine the output file path
            output_file_path = os.path.join(args.output_dir, os.path.basename(args.input_path))
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Save the optimized image
            print(f"Saving optimized NIfTI to: {output_file_path}")
            nib.save(optimized_img, output_file_path)
            
            print(f"Optimized NIfTI file created successfully at: {output_file_path}")
        else:
            print(f"Error: Input path {args.input_path} is neither a valid directory nor a NIfTI file.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 