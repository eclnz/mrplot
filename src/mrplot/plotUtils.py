import matplotlib.pyplot as plt  # type: ignore
import imageio  # type: ignore
import os
from typing import Optional, List
import nibabel as nib  # type: ignore
from dataclasses import dataclass
from mrplot.MRISlices import MRISlices
from mrplot.plotConfig import PlotConfig
from mrplot.indexingUtils import list_bids_subjects_sessions_scans, build_series_list
from pathlib import Path
import importlib.resources
import numpy as np
import logging

plt.switch_backend("Agg")  # For non-GUI environments


class MRIDataProcessor:
    """Handles preprocessing of MRI data, underlay images, and masks."""

    def __init__(
        self,
        mri_data_path: str,
        config: PlotConfig,
        underlay_image_path: Optional[str] = None,
        mask_path: Optional[str] = None,
    ):
        self.mri_data_path = mri_data_path
        self.underlay_image_path = underlay_image_path
        self.mask_path = mask_path
        self.config = config
        self._import_scans()
        self._extract_slices()
        self._get_media_type()

    def _import_scans(self):
        self.mri_data = nib.load(self.mri_data_path)
        self.mask = nib.load(self.mask_path) if self.mask_path else None
        self.underlay_image = (
            nib.load(self.underlay_image_path) if self.underlay_image_path else None
        )

    def _extract_slices(self):
        """Extract the middle slices of volumes"""

        # Check if the underlay matches spatial dimensions if present.
        if self.underlay_image is not None:
            if self.underlay_image.shape[:3] != self.mri_data.shape[:3]:
                raise ValueError(
                    "Size of underlay image does not match size of MRI data in the first three dimensions"
                )

        # If mask present
        if self.mask is not None:
            # Check if the mask matches spatial dimensions if present.
            if self.mask.shape[:3] != self.mri_data.shape[:3]:
                raise ValueError(
                    "Size of mask does not match size of MRI data in the first three dimensions"
                )

            # User specifies to crop the image
            if self.config.crop is True:
                # Read MRI with the mask applied
                self.mri_slices = MRISlices.from_nibabel(
                    self.config, self.mri_data, self.mask, apply_mask=True
                )

                # If the user also supplies underlay read in
                if self.underlay_image is not None:
                    # If the user specifies for underlay to be masked
                    if self.config.mask_underlay:
                        self.underlay_slices = MRISlices.from_nibabel(
                            self.config, self.underlay_image, self.mask, apply_mask=True
                        )
                    else:
                        self.underlay_slices = MRISlices.from_nibabel(
                            self.config,
                            self.underlay_image,
                            self.mask,
                            apply_mask=False,
                        )
                # If user does not supply underlay its set to None
                else:
                    self.underlay_slices = None
            # If user provides mask but does not set crop to true
            else:
                self.mri_slices = MRISlices.from_nibabel(
                    self.config, self.mri_data, self.mask, apply_mask=False
                )
                self.underlay_slices = MRISlices.from_nibabel(
                    self.config, self.underlay_image, self.mask, apply_mask=False
                )

        # If user does not provide a mask at all
        else:
            # Read mri image normally
            self.mri_slices = MRISlices.from_nibabel(self.config, self.mri_data)

            # If user provides underlay, then read that in without masking
            if self.underlay_image is not None:
                self.underlay_slices = MRISlices.from_nibabel(
                    self.config, self.underlay_image
                )
            # If no underlay supplied its set to none.
            else:
                self.underlay_slices = None

        # Make sure the images are rotated from RSL so they appear normal clinically
        self.mri_slices.rotate_slices()
        if self.underlay_image is not None:
            self.underlay_slices.rotate_slices()

    def _get_media_type(self) -> None:
        slice_dims = len(self.mri_slices.axial.shape)
        if slice_dims == 2:
            self.media_type = "png"
        elif slice_dims > 2:
            self.media_type = "mp4"

    def process_scan(self, scan_path: str, config: PlotConfig) -> None:
        """Process a scan and generate plots/videos"""
        try:
            img = nib.load(scan_path)
            data = img.get_fdata()
            
            # Handle different dimensionalities
            if data.ndim == 3:
                self._process_3d(data, config)
            elif data.ndim == 4:
                self._process_4d(data, config)
            elif data.ndim == 5:
                self._process_5d(data, config)
            else:
                raise ValueError(f"Unsupported data dimensionality: {data.ndim}D")
                
        except Exception as e:
            logging.error(f"Failed to process {scan_path}: {str(e)}")
            raise

    def _process_4d(self, data: np.ndarray, config: PlotConfig) -> None:
        """Process 4D data (time series) and create videos"""
        # Average over time dimension
        mean_data = np.mean(data, axis=-1)
        
        # Create orthogonal plots
        self._create_orthogonal_plots(mean_data, config)
        
        # Create video only if there are multiple time points
        if data.shape[-1] > 1:
            self._create_video(data, config)
        else:
            logging.info(f"Skipping video creation for {config.scan_name} - single time point")

    def _create_video(self, data: np.ndarray, config: PlotConfig) -> None:
        """Create an animated video from 4D data"""
        from matplotlib.animation import FFMpegWriter
        
        fig = plt.figure(figsize=(10, 8))
        writer = FFMpegWriter(fps=10)
        
        output_path = Path(config.output_dir) / f"{config.scan_name}_{config.plane}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with writer.saving(fig, str(output_path), dpi=100):
            for t in range(data.shape[-1]):
                self._plot_frame(data[..., t], config, fig)
                writer.grab_frame()
                plt.clf()
        
        plt.close()


class MRIPlotter:
    """Handles plotting of MRI data, generating either images or videos based on the media type."""

    def __init__(
        self,
        media_type: str,
        mri_data: MRISlices,
        config: PlotConfig,
        output_dir: str,
        scan_name: str,
        underlay_image: Optional[MRISlices] = None,
    ):
        """
        Args:
            media_type (str): File format for the output (e.g., 'png', 'mp4').
            mri_data (MRISlices): MRI data to be plotted.
            config (PlotConfig): Configuration settings for plotting.
            output_dir (str): Directory to save the output.
            scan_name (str): Base name for the output files.
            underlay_image (Optional[MRISlices]): Underlay image for overlays.
        """
        self.media_type = media_type
        self.mri_data = mri_data
        self.config = config
        self.output_dir = output_dir
        self.scan_name = scan_name
        self.underlay_image = underlay_image

        os.makedirs(self.output_dir, exist_ok=True)

    def plot(self):
        """Plots the MRI data based on the specified media type."""
        if self.media_type == "png":
            self._plot_images()
        elif self.media_type == "mp4":
            self._plot_videos()
        else:
            raise ValueError(
                f"Unsupported media type: {self.media_type}. Only 'png' and 'mp4' are supported."
            )

    def _plot_images(self):
        """Generates and saves images for three orthogonal planes."""
        plane_images = self.mri_data.add_titles_and_generate_images(
            config=self.config, title_prefix=self.scan_name, single_timepoint=True
        )

        for plane, image in plane_images.items():
            output_path = os.path.join(self.output_dir, f"{self.scan_name}_{plane}.png")
            imageio.imwrite(output_path, image)

    def _plot_videos(self):
        """Generates videos for three orthogonal planes."""
        video_writers = self._initialize_video_writers()

        # Iterate over timepoints
        for t in range(self.mri_data.shape[-1]):  # Assumes last dimension is timepoint
            plane_images = self.mri_data.add_titles_and_generate_images(
                config=self.config,
                title_prefix=self.scan_name,
                single_timepoint=False,
                slice_timepoint=t,
                underlay_slice=self.underlay_image,
            )

            # Write all frames by appending images to video writers
            for plane, image in plane_images.items():
                video_writers[plane].append_data(image)

        # Close all video writers
        for writer in video_writers.values():
            writer.close()

    def _initialize_video_writers(self):
        """Initializes video writers for each plane."""
        return {
            plane: imageio.get_writer(
                os.path.join(self.output_dir, f"{self.scan_name}_{plane}.mp4"),
                fps=self.config.fps,
                codec="libx264",
            )
            for plane in ["sagittal", "coronal", "axial"]
        }


def find_unique_scans_in_bids(
    bids_folder: str, file_extension: str = ".nii.gz"
) -> List[str]:
    """
    Scans a BIDS folder and identifies unique scans based on text after `desc-` in filenames.

    Args:
        bids_folder (str): Path to the BIDS folder to scan.
        file_extension (str): File extension to look for (default: '.nii.gz').

    Returns:
        List[str]: A sorted list of unique scan descriptions found in the BIDS folder.
    """

    if not os.path.isdir(bids_folder):
        raise ValueError(
            f"BIDS folder '{bids_folder}' does not exist or is not a directory."
        )

    subject_session = list_bids_subjects_sessions_scans(
        bids_folder, file_extension=".nii.gz"
    )

    unique_scans = build_series_list(subject_session)

    return unique_scans


@dataclass
class GroupPlotConfig:
    """Configuration for group plotting."""

    bids_dir: str
    output_dir: str
    selected_scans: list[str]
    all_scans: list[str]


class GroupPlotter:
    def __init__(
        self,
        config: GroupPlotConfig,
        subject_session_list: dict[str, dict[str, dict[str, dict[str, str]]]],
    ):
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

    def _find_scan_path(
        self, subject: str, session: str, scan_name: str | None
    ) -> Optional[str]:
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
                        mask_path = self._find_scan_path(
                            subject, session, scan_config.mask
                        )
                        underlay_path = self._find_scan_path(
                            subject, session, scan_config.underlay_image
                        )

                        # Initialize and preprocess the MRI data
                        processor = MRIDataProcessor(
                            mri_data_path=metadata["scan_path"],
                            config=scan_config,
                            underlay_image_path=underlay_path,
                            mask_path=mask_path,
                        )

                        # Initialize and run the plotter
                        plotter = MRIPlotter(
                            media_type=processor.media_type,
                            mri_data=processor.mri_slices,
                            config=scan_config,
                            output_dir=os.path.join(
                                self.config.output_dir, subject, session
                            ),
                            scan_name=scan,
                            underlay_image=processor.underlay_slices,
                        )
                        plotter.plot()

                        print(f"successfully plotted {scan}")

                    except Exception as e:
                        print(f"Error plotting {scan}: {e}")


def get_default_config():
    config_path = importlib.resources.files("mrplot") / "configs/default.json"
    return load_config(config_path)
