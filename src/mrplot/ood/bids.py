import os
import json
import nibabel as nib
import re
import numpy as np
import shutil
from typing import List, Union, Optional, Tuple, Dict, cast, Any
from mrplot.ood.slice import Slice
from mrplot.ood.print import print_tree
from mrplot.ood.clogging import logger  # Import the logger
import sys # Import sys module
import logging
import pickle

def validate_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

class BIDS:
    def __init__(self, path: str):
        validate_path(path)
        self.path = path
        self.subjects: List[Subject] = []
        self.raw_path = self._raw_folder(path)
        self._load_subjects()
        self._load_derivatives()

    def _raw_folder(self, path: str) -> str:
        raw_path = os.path.join(path, "raw")
        if not raw_path:
            raw_path = path
        return raw_path

    def _load_subjects(self) -> None:
        for subject_name in os.listdir(self.raw_path):
            if not subject_name.startswith("sub-"):
                continue
            subject = Subject(os.path.join(self.raw_path, subject_name))
            self.subjects.append(subject)
        if not self.subjects:
            raise FileNotFoundError(f"No subjects found in {self.raw_path}")

    def _load_derivatives(self) -> None:
        derivatives_path = os.path.join(self.path, "derivatives")
        if not os.path.exists(derivatives_path):
            return

        for derivative_name in os.listdir(derivatives_path):
            derivative_dir = os.path.join(derivatives_path, derivative_name)
            if not os.path.isdir(derivative_dir):
                continue

            for subject in self.subjects:
                subject_deriv_dir = os.path.join(derivative_dir, subject.get_name())
                if not os.path.exists(subject_deriv_dir):
                    continue

                for session in subject.sessions:
                    try:
                        session_deriv_dir = os.path.join(
                            subject_deriv_dir, session.get_name()
                        )
                        if not os.path.exists(session_deriv_dir):
                            continue

                        session.load_scan_types(session_deriv_dir)
                    except Exception as e:

                        logger.error(
                            f"Error loading derivatives for {subject.get_name()}/{session.get_name()}: {str(e)}",
                            exc_info=True  # Include exception info in the log
                        )

    def print_tree(self, include_details: bool = False) -> None:
        """
        Print a visual tree representation of the BIDS dataset structure.

        Args:
            include_details: If True, include additional details like paths
        """
        nodes = []
        for subject in self.subjects:
            nodes.append((0, str(subject)))

            for session in subject.sessions:
                nodes.append((1, str(session)))

                for scan in session.scans:
                    if include_details:
                        scan_info = f"{scan} ({scan.path})"
                    else:
                        scan_info = str(scan)
                    nodes.append((2, scan_info))

        print_tree(nodes)

    def create_slices(
        self,
        scan_regex: str,
        origin_regex: str,
        required_dims: Optional[int] = None,
        slice_indices: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, Slice]:
        logger.info(f"Creating slices with scan_regex='{scan_regex}', origin_regex='{origin_regex}'") # Info log
        slices: Dict[str, Slice] = {}
        for subject in self.subjects:
            logger.debug(f"Processing subject: {subject.get_name()}") # Debug log
            for session in subject.sessions:
                logger.debug(f"Processing session: {session.get_name()}") # Debug log
                for scan in session.scans:
                    if re.match(scan_regex, scan.scan_name) and re.match(origin_regex, scan.origin):
                        logger.debug(f"Found matching scan: {scan.path}") # Debug log
                        if required_dims is not None:
                            if len(scan.shape) != required_dims:
                                logger.debug(f"Skipping scan {scan.path} due to incorrect dimensions ({len(scan.shape)} != {required_dims})") # Debug log
                                continue
                        
                        logger.debug(f"Loading NIfTI data for: {scan.path}") # Debug log for slow process
                        data = nib.load(scan.path).get_fdata() # type: ignore
                        logger.debug(f"Successfully loaded data for: {scan.path}") # Debug log
                        slice_obj = Slice(slice_indices)
                        identifier = f"{subject.get_name()}_{session.get_name()}_{scan.get_name()}"
                        slice_obj.load_slices(data)
                        slices[identifier] = slice_obj
                        logger.debug(f"Created and added slice for: {scan.path}") # Debug log
        if not slices:
            # Keep this as a ValueError, as it indicates a potential configuration issue
            raise ValueError(f"No scans found matching scan_regex='{scan_regex}' and origin_regex='{origin_regex}'")
        logger.info(f"Created {len(slices)} slices successfully.") # Info log
        return slices

    def create_optimized_dataset(
        self,
        output_path: str,
        scan_regex: str = ".*",
        origin_regex: str = ".*",
        central_slices_only: bool = True,
        slice_offsets: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Creates an optimized version of the BIDS dataset with only specific slices.
        
        Args:
            output_path: Path where the optimized dataset will be saved
            scan_regex: Regular expression to match scan names
            origin_regex: Regular expression to match scan origins
            central_slices_only: If True, use the central slices or offsets in each dimension
            slice_offsets: Dictionary with relative offsets from center for each plane
                           e.g., {'sagittal': 0.1, 'coronal': -0.05, 'axial': 0.2}
                           Values are relative to image dimensions (-0.5 to 0.5),
                           where 0 is center, -0.5 is start, and 0.5 is end
        """
        logger.info(f"Creating optimized dataset at {output_path} with scan_regex='{scan_regex}', origin_regex='{origin_regex}'")
        
        if slice_offsets is None:
            slice_offsets = {'sagittal': 0.0, 'coronal': 0.0, 'axial': 0.0}
        else:
            for plane in ['sagittal', 'coronal', 'axial']:
                if plane not in slice_offsets:
                    slice_offsets[plane] = 0.0
                else:
                    # Validate offset is within valid range (-0.5 to 0.5)
                    if slice_offsets[plane] < -0.5 or slice_offsets[plane] > 0.5:
                        logger.warning(f"Offset for {plane} plane is outside valid range (-0.5 to 0.5). Using {max(-0.5, min(0.5, slice_offsets[plane]))}.")
                        slice_offsets[plane] = max(-0.5, min(0.5, slice_offsets[plane]))
        
        logger.info(f"Using slice offsets: sagittal={slice_offsets['sagittal']}, coronal={slice_offsets['coronal']}, axial={slice_offsets['axial']}")
        
        if os.path.exists(output_path):
            logger.warning(f"Output path {output_path} already exists. Files may be overwritten.")
        else:
            os.makedirs(output_path, exist_ok=True)
        
        for subject in self.subjects:
            logger.debug(f"Processing subject: {subject.get_name()}")
            for session in subject.sessions:
                logger.debug(f"Processing session: {session.get_name()}")
                for scan in session.scans:
                    if re.match(scan_regex, scan.scan_name) and re.match(origin_regex, scan.origin):
                        logger.debug(f"Processing scan: {scan.path}")
                        
                        # Load the original NIfTI file
                        logger.debug(f"Loading NIfTI file: {scan.path}")
                        nifti_img = cast(nib.Nifti1Image, nib.load(scan.path))
                        data = nifti_img.get_fdata()
                        
                        # Create a new array with zeros
                        if central_slices_only:
                            # Create an array of zeros with the same shape as the original
                            optimized_data = np.zeros_like(data)
                            
                            # Calculate slice indices based on offsets
                            # Convert relative offsets (-0.5 to 0.5) to actual indices
                            x_idx = int((0.5 + slice_offsets['sagittal']) * (data.shape[0] - 1))
                            y_idx = int((0.5 + slice_offsets['coronal']) * (data.shape[1] - 1))
                            z_idx = int((0.5 + slice_offsets['axial']) * (data.shape[2] - 1))
                            
                            # Ensure indices are within bounds
                            x_idx = max(0, min(data.shape[0] - 1, x_idx))
                            y_idx = max(0, min(data.shape[1] - 1, y_idx))
                            z_idx = max(0, min(data.shape[2] - 1, z_idx))
                            
                            logger.debug(f"Using slices at: sagittal={x_idx}/{data.shape[0]-1}, coronal={y_idx}/{data.shape[1]-1}, axial={z_idx}/{data.shape[2]-1}")
                            
                            # Copy only the specified slices
                            optimized_data[x_idx, :, :] = data[x_idx, :, :]  # Sagittal slice
                            optimized_data[:, y_idx, :] = data[:, y_idx, :]  # Coronal slice
                            optimized_data[:, :, z_idx] = data[:, :, z_idx]  # Axial slice
                        else:
                            # Use the full data (for testing or comparison)
                            optimized_data = data
                        
                        # Create the optimized NIfTI image with the same header
                        optimized_img = nib.Nifti1Image(
                            optimized_data, 
                            nifti_img.affine, 
                            nifti_img.header
                        )
                        
                        # Determine the output path structure
                        # Preserve the original directory structure relative to the BIDS root
                        rel_path = os.path.relpath(scan.path, self.path)
                        output_file_path = os.path.join(output_path, rel_path)
                        
                        # Create directories if they don't exist
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        
                        # Save the optimized image
                        logger.debug(f"Saving optimized NIfTI to: {output_file_path}")
                        nib.save(optimized_img, output_file_path)
                        
                        # Copy the JSON sidecar if it exists
                        json_sidecar_path = scan.path.replace(".nii.gz", ".json").replace(".nii", ".json")
                        if os.path.exists(json_sidecar_path):
                            output_json_path = output_file_path.replace(".nii.gz", ".json").replace(".nii", ".json")
                            shutil.copy(json_sidecar_path, output_json_path)
                            logger.debug(f"Copied JSON sidecar to: {output_json_path}")
        
        logger.info(f"Optimized dataset created at: {output_path}")

class Subject:
    def __init__(self, path: str):
        validate_path(path)
        self.path = path
        self.subject_id = os.path.basename(path).split("-")[1]
        self.sessions: List[Session] = []
        self.load_sessions(path)

    def __repr__(self) -> str:
        return f"Subject(id={self.subject_id})"

    def load_sessions(self, path: str) -> None:
        validate_path(path)
        for session_name in os.listdir(path):
            if not session_name.startswith("ses-"):
                continue
            session = Session(os.path.join(path, session_name))
            self.sessions.append(session)
        if not self.sessions:
            raise FileNotFoundError(f"No sessions found in {path}")

    def get_id(self) -> str:
        return self.subject_id

    def get_name(self) -> str:
        return f"sub-{self.subject_id}"


class Session:
    def __init__(self, path: str):
        validate_path(path)
        self.path = path
        self.session_id = os.path.basename(path).split("-")[1]
        self.scans: List[Scan] = []
        self.load_scan_types(path)

    def __repr__(self) -> str:
        return f"Session(id={self.session_id})"

    def load_scan_types(self, path: str) -> None:
        if not os.path.exists(path):
            return

        try:
            scan_type_dirs = [
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            ]
            if not scan_type_dirs:
                self.load_scans(path)
                return

            for scan_type in scan_type_dirs:
                scan_type_path = os.path.join(path, scan_type)
                try:
                    self.load_scans(scan_type_path)
                except Exception as e:
                    # Use logger.error
                    logger.error(f"Error loading scans from {scan_type_path}: {str(e)}", exc_info=True)
        except Exception as e:
            # Use logger.error
            logger.error(f"Error accessing directory {path}: {str(e)}", exc_info=True)

    def load_scans(self, path: str) -> None:
        if not os.path.exists(path):
            return

        try:
            scan_files = [
                f
                for f in os.listdir(path)
                if f.endswith(".nii.gz") or f.endswith(".nii")
            ]
            if not scan_files:
                return

            for scan_name in scan_files:
                scan_path = os.path.join(path, scan_name)
                try:
                    scan = Scan(scan_path)
                    self.add_scan(scan)
                except Exception as e:
                    # Use logger.error
                    logger.error(f"Error loading scan {scan_path}: {str(e)}", exc_info=True)
        except Exception as e:
            # Use logger.error
            logger.error(f"Error listing directory {path}: {str(e)}", exc_info=True)

    def add_scan(self, scan: "Scan") -> None:
        if scan not in self.scans:
            self.scans.append(scan)
        else:
            # Use logger.warning for non-critical issues
            logger.warning(f"Scan {scan} already exists in {self.get_name()}")

    def get_id(self) -> str:
        return self.session_id

    def get_name(self) -> str:
        return f"ses-{self.session_id}"


class Scan:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scan file not found: {path}")

        self.path = path
        logger.debug(f"Initializing Scan object for: {path}") # Debug log
        self.scan_name = self._get_scan_name()
        self.origin = self._get_origin()
        self.shape = self._get_shape()

        try:
            self.json_sidecar = self._load_json_sidecar()
        except Exception as e:
            # Use logger.warning for non-critical loading issues
            logger.warning(f"Failed to load JSON sidecar for {path}: {str(e)}", exc_info=True)
            self.json_sidecar = None

    def __repr__(self) -> str:
        return f"Scan({self.scan_name}) - {self.origin}"

    def _get_scan_name(self) -> str:
        if len(os.path.basename(self.path).split("_")) > 2:
            name = "_".join(os.path.basename(self.path).split("_")[2:])
        else:
            name = os.path.basename(self.path)
        if ".nii.gz" in name:
            name = name.split(".nii.gz")[0]
        elif ".nii" in name:
            name = name.split(".nii")[0]
        return name

    def _get_origin(self) -> str:
        path_parts = self.path.split(os.sep)
        if "derivatives" in path_parts:
            derivatives_index = path_parts.index("derivatives")
            if len(path_parts) > derivatives_index + 1:
                return path_parts[derivatives_index + 1]
            return "derivatives"
        elif "raw" in path_parts:
            return "raw"
        return "unknown"

    def _get_shape(self) -> Tuple[int, int, int]:
        try:
            logger.debug(f"Attempting to load shape for: {self.path}") # Debug log
            shape = nib.load(self.path).shape  # type: ignore
            logger.debug(f"Successfully loaded shape {shape} for: {self.path}") # Debug log
            return shape
        except Exception as e:
            logger.error(f"Failed to get shape of {self.path}: {str(e)}", exc_info=True) # Keep error log
            raise RuntimeError(f"Failed to get shape of {self.path}: {str(e)}")

    def _load_json_sidecar(self) -> Union[dict, None]:
        json_path = self.path.replace(".nii.gz", ".json").replace(".nii", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON in sidecar file {json_path}: {str(e)}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON sidecar {json_path}: {str(e)}")
        return None

    def get_name(self) -> str:
        return self.scan_name


# if __name__ == "__main__":
#     logger.setLevel(logging.DEBUG)

#     bids = BIDS("/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4")
#     bids.print_tree()
    
#     # Example 2: Create optimized dataset with specific offsets for each plane
#     # For example, to target lateral ventricles which might be slightly above center in axial view
#     # and slightly posterior to center in sagittal view
#     bids.create_optimized_dataset(
#         output_path="/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4_optimized_ventricles",
#         scan_regex=".*",
#         origin_regex=".*",
#         slice_offsets={
#             'sagittal': 0.0,    # Center in sagittal plane (left-right)
#             'coronal': -0.1,    # Slightly anterior in coronal plane (10% toward front from center)
#             'axial': 0.15       # Slightly above center in axial plane (15% above center)
#         }
#     )
