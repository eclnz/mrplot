from dataclasses import dataclass
from typing import Optional
import os
import importlib
from mrplot.plotUtils import PlotConfig, MRIDataProcessor, MRIPlotter

@dataclass
class GroupPlotConfig:
    """Configuration for group plotting."""
    bids_dir: str
    output_dir: str
    selected_scans: list[str]
    all_scans: list[str]

class GroupPlotter:
    def __init__(self, config: GroupPlotConfig, subject_session_list: dict[str, dict[str, dict[str, dict[str, str]]]]):
        """
        Initializes the GroupPlotter.

        Args:
            config (GroupPlotConfig): Configuration for the group plotting session.
            subject_session_list (Dict): Dictionary of subjects, sessions, and scans.
        """
        self.config = config
        self.subject_session_list = subject_session_list
        self.scan_configs = self._initialize_scan_configs()

    def _initialize_scan_configs(self) -> dict[str, PlotConfig]:
        """
        Initializes scan options by prompting the user to configure settings for each scan.

        Returns:
            dict: A dictionary mapping scan names to PlotConfig objects.
        """
        configs = {}
        for scan in self.config.selected_scans:
            configs[scan] = PlotConfig()
        return configs
        
    def _find_scan_path(self, subject: str, session: str, scan_name: str|None) -> Optional[str]:
        """
        Finds the path to a scan (e.g., mask or underlay) within the subject-session structure.

        Args:
            subject (str): Subject ID.
            session (str): Session ID.
            scan_name (str): Name of the scan to locate.

        Returns:
            Optional[str]: The path to the scan if found, otherwise None.
        """
        session_data = self.subject_session_list.get(subject, {}).get(session, {})
        for scan, scan_metadata in session_data.items():
            if scan_name and scan_name in scan:
                return scan_metadata["scan_path"]
        return None

    def _select_scan(self, prompt: str) -> Optional[str]:
        """
        Prompts the user to select a scan.

        Args:
            prompt (str): Prompt message.

        Returns:
            Optional[str]: Selected scan or None.
        """
        print("Available scans:")
        for idx, scan in enumerate(self.config.all_scans, 1):
            print(f"{idx}: {scan}")
        while True:
            choice = input(f"{prompt} (enter number or leave blank for none): ").strip()
            if not choice:
                return None
            try:
                index = int(choice) - 1
                if 0 <= index < len(self.config.all_scans):
                    return self.config.all_scans[index]
                print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid input. Enter a number.")

    def plot(self):
        """
        Plots the selected scans using configured options.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        for subject, sessions in self.subject_session_list.items():
            for session, scans in sessions.items():
                for scan, metadata in scans.items():
                    if scan not in self.config.selected_scans:
                        continue

                    try:
                        # Retrieve paths for mask and underlay
                        scan_config = self.scan_configs[scan]
                        mask_path = self._find_scan_path(subject, session, scan_config.mask)
                        underlay_path = self._find_scan_path(subject, session, scan_config.underlay_image)

                        # Initialize and preprocess the MRI data
                        processor = MRIDataProcessor(
                            mri_data_path=metadata["scan_path"],
                            config=scan_config,
                            underlay_image_path=underlay_path,
                            mask_path=mask_path
                        )

                        # Initialize and run the plotter
                        plotter = MRIPlotter(
                            media_type=processor.media_type,
                            mri_data=processor.mri_slices,
                            config=scan_config,
                            output_dir=os.path.join(self.config.output_dir, subject, session),
                            scan_name=scan,
                            underlay_image=processor.underlay_slices
                        )
                        plotter.plot()
                        
                        print(f"successfully plotted {scan}")

                    except Exception as e:
                        print(f"Error plotting {scan}: {e}")

def get_default_config():
    config_path = importlib.resources.files("mrplot") / "configs/default.json"
    return load_config(config_path)