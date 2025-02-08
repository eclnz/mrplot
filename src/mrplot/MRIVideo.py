import os
from typing import Dict

class MRIMedia:
    def __init__(self, output_dir: str, scan_name: str, media_type: str):
        """
        Initialize the MRIMedia class.

        Args:
            output_dir (str): Directory where media files will be stored.
            scan_name (str): Base name for the scan.
            file_format (str): File extension (e.g., 'mp4', 'gif').
        """
        self.output_dir = output_dir
        self.scan_name = scan_name
        self.file_format = media_type
        self.media_paths = self._get_media_paths()

    def _get_media_paths(self) -> Dict[str, str]:
        """
        Generate paths for each view.

        Returns:
            Dict[str, str]: A dictionary mapping views to their file paths.
        """
        views = ['sagittal', 'coronal', 'axial']
        media_paths = {}
        for view in views:
            file_name = f"{self.scan_name}_{view}.{self.file_format}"
            media_paths[view] = os.path.join(self.output_dir, file_name)
        return media_paths