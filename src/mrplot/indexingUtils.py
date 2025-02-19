from typing import Dict
from collections import defaultdict
from pathlib import Path
import os
import logging

BIDS_RAW_OPTIONS = ["anat", "func", "fmap", "dwi", "perf"]


def list_bids_subjects_sessions_scans(
    data_directory: str, file_extension: str
) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Recursively traverses directories to list files by subject, session, and scan in a BIDS-compliant structure.

    Args:
        data_directory (str): Path to the base directory containing files.
        file_extension (str): File extension to look for (e.g., '.nii.gz').

    Returns:
        Dict[str, Dict[str, Dict[str, Dict[str, str]]]]: A dictionary with subjects, sessions, and scans containing metadata.
    """
    subjects_sessions_scans: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    data_path = Path(data_directory)

    # 1. Check directory existence and type first
    if not data_path.exists():
        raise ValueError(f"Data directory '{data_directory}' does not exist")
    if not data_path.is_dir():
        raise ValueError(f"Path '{data_directory}' is not a directory")

    # 2. Then check for BIDS structure
    def has_valid_structure(entry: Path) -> bool:
        if entry.name == "derivatives":
            return True
        if entry.name.startswith("sub-"):
            return True
        if entry.parent.name.startswith("sub-") and entry.name.startswith("ses-"):
            return True
        return False

    if not any(has_valid_structure(entry) for entry in data_path.iterdir()):
        raise ValueError(f"No valid BIDS structure found in {data_directory}")

    def recursive_traverse(path: Path):
        """
        Recursively traverses the directory structure to detect subjects, sessions, and scans.

        Args:
            path (Path): Current directory path to process.
        """
        for entry in path.iterdir():
            if entry.is_dir():
                # Handle subject directories
                if entry.name.startswith("sub-"):
                    recursive_traverse(entry)  # Process sessions within the subject

                # Handle session directories
                elif entry.name.startswith("ses-"):
                    subject_dir = entry.parent.name
                    if subject_dir.startswith("sub-"):
                        recursive_traverse(entry)  # Process scans within the session

                # Traverse deeper for other directories
                else:
                    recursive_traverse(entry)

            elif entry.is_file() and (
                entry.name.endswith(file_extension) or file_extension in entry.name
            ):
                # Extract metadata
                if entry.parent.name in BIDS_RAW_OPTIONS:
                    parent_session = entry.parent.parent.name
                    parent_subject = entry.parent.parent.parent.name
                else:
                    parent_session = entry.parent.name
                    parent_subject = entry.parent.parent.name

                # Ensure the hierarchy is valid
                if not parent_subject.startswith("sub-"):
                    parent_subject = "unknown"  # Fallback for subject

                if not parent_session.startswith("ses-"):
                    parent_session = "unknown"  # Fallback for session

                # Skip files that cannot be matched to a valid subject/session structure
                if parent_subject == "unknown" or parent_session == "unknown":
                    logging.warning(
                        f"Skipping file {entry.name} due to invalid subject/session structure"
                    )
                    continue

                # Extract scan description
                parts = entry.name.split("_desc-")
                if len(parts) > 1:
                    scan = parts[1]
                else:
                    scan = entry.name

                # Populate the structure
                subjects_sessions_scans[parent_subject][parent_session][scan][
                    "scan_path"
                ] = os.path.join(path, entry.name)

    # Start recursive traversal
    recursive_traverse(data_path)

    # Convert defaultdict to standard dictionary for cleaner return
    result = {
        k: {kk: dict(vv) for kk, vv in v.items()}
        for k, v in subjects_sessions_scans.items()
    }
    
    # Check if any scans were found
    if not result:
        raise ValueError(f"No scans found in BIDS directory: {data_directory}")
        
    return result


def build_series_list(
    subjects_sessions_scans: dict[str, dict[str, dict[str, dict[str, str]]]],
) -> list:
    """
    Finds all unique scan names in the subjects_sessions_scans structure.

    Args:
        subjects_sessions_scans (Dict): The nested structure of subjects, sessions, and scans.

    Returns:
        set: A set of unique scan names.
    """
    unique_scans = set()

    for subject_id, sessions in subjects_sessions_scans.items():
        for session_id, scans in sessions.items():
            for scan in scans:
                if scan != "cohort":  # Ignore the cohort key
                    unique_scans.add(scan)

    return sorted(unique_scans)
