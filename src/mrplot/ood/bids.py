import os
import json
import nibabel as nib
import re
from typing import List, Union, Optional, Tuple
from mrplot.ood.slice import Slice, SliceCollection
from mrplot.ood.print import print_tree


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
                        print(
                            f"Error loading derivatives for {subject.get_name()}/{session.get_name()}: {str(e)}"
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
        self, scan_regex: str, origin_regex: str, required_dims: Optional[int] = None, slice_indices: Optional[Tuple[int, int, int]] = None
    ) -> SliceCollection:
        slices = []
        for subject in self.subjects:
            for session in subject.sessions:
                for scan in session.scans:
                    if re.match(scan_regex, scan.scan_name) and re.match(origin_regex, scan.origin):
                        if required_dims is not None:
                            if len(scan.shape) != required_dims:
                                continue
                        slice = Slice(
                            subject_id=subject.get_id(),
                            session_id=session.get_id(),
                            scan_id=scan.get_name(),
                            path=scan.path,
                            slice_indices=slice_indices,
                            origin=scan.origin,
                        )
                        slices.append(slice)
        if not slices:
            raise ValueError(f"No scans found matching regex '{scan_regex}'")
        return SliceCollection(slices)


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
                    print(f"Error loading scans from {scan_type_path}: {str(e)}")
        except Exception as e:
            print(f"Error accessing directory {path}: {str(e)}")

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
                    print(f"Error loading scan {scan_path}: {str(e)}")
        except Exception as e:
            print(f"Error listing directory {path}: {str(e)}")

    def add_scan(self, scan: "Scan") -> None:
        if scan not in self.scans:
            self.scans.append(scan)
        else:
            print(f"Scan {scan} already exists in {self.get_name()}")

    def get_id(self) -> str:
        return self.session_id

    def get_name(self) -> str:
        return f"ses-{self.session_id}"


class Scan:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scan file not found: {path}")

        self.path = path
        self.scan_name = self._get_scan_name()
        self.origin = self._get_origin()
        self.shape = self._get_shape()

        try:
            self.json_sidecar = self._load_json_sidecar()
        except Exception as e:
            print(f"Warning: Failed to load JSON sidecar for {path}: {str(e)}")
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
            return nib.load(self.path).shape #type: ignore
        except Exception as e:
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
