import nibabel as nib
from typing import Optional, Tuple, List, Dict, Any, Set, Union
import numpy as np
import re
import gc
from enum import Enum, auto
from dataclasses import dataclass
from mrplot.ood.plotting import ImageLayer, VectorFieldLayer, MaskContourLayer
from mrplot.ood.plotting import PlotComposer
from mrplot.ood.print import print_tree


def reorient_slice(slice_data: np.ndarray) -> np.ndarray:
    return np.rot90(slice_data)


class SliceType(Enum):
    T1 = auto()
    T2 = auto()
    BOLD = auto()
    MASK = auto()
    MOTION = auto()
    FIELDMAP = auto()


@dataclass
class SliceRelation:
    source_type: SliceType
    target_type: SliceType
    relation_name: str


class Slice:
    def __init__(
        self,
        slice_indices: Optional[Tuple[int, int, int]] = None,
    ):
        self.slice_indices = slice_indices
        self.sagittal = None
        self.coronal = None
        self.axial = None

    def load_slices(
        self,
        data: np.ndarray,
    ):
        self.sagittal, self.coronal, self.axial = self._load_slices(data)

    def _load_slices(
        self,
        data: np.ndarray,
    ):
        if data is not None:
            volume = np.asarray(data).astype(np.float32)
        else:
            raise ValueError("Either path or data must be provided.")

        if self.slice_indices is None:
            self.slice_indices = (
                volume.shape[0] // 2,
                volume.shape[1] // 2,
                volume.shape[2] // 2,
            )

        # Create slices - make copies to ensure they persist when volume is deleted
        sagittal = volume[self.slice_indices[0], :, :, ...].copy()
        coronal = volume[:, self.slice_indices[1], :, ...].copy()
        axial = volume[:, :, self.slice_indices[2], ...].copy()
        
        del volume
        gc.collect()
        
        return sagittal, coronal, axial



    def set_view(self, view_type: str) -> None:
        if view_type == "sagittal":
            self.current = self.sagittal
            self.current_view = "sagittal"
        elif view_type == "coronal":
            self.current = self.coronal
            self.current_view = "coronal"
        elif view_type == "axial":
            self.current = self.axial
            self.current_view = "axial"
        else:
            raise ValueError(
                "Invalid view type. Must be 'sagittal', 'coronal', or 'axial'."
            )

    def to_vector_layer(
        self, thin_factor: int = 6, scale: float = 10.0
    ) -> VectorFieldLayer:
        """Create a vector field layer from the current slice."""
        if self.current is None:
            raise ValueError("No slice data available. Call load_slices first.")
            
        slice_data = self.current

        # Handle 4D data (vector field over time)
        if len(slice_data.shape) == 4:
            time_point = 0
            if slice_data.shape[3] > 1:
                time_point = 1
            vector_data = slice_data[..., time_point]
            return VectorFieldLayer(
                np.flip(vector_data, axis=0),
                slice_type=self.current_view,
                thin_factor=thin_factor,
                scale=scale,
            )

        # Handle 3D data with vector components
        elif len(slice_data.shape) == 3 and slice_data.shape[2] >= 3:
            vector_components = slice_data[..., :3]
            return VectorFieldLayer(
                vector_components,
                slice_type=self.current_view,
                thin_factor=thin_factor,
                scale=scale,
            )

        else:
            raise ValueError(
                "Current slice data cannot be converted to a vector field layer. "
                "Expected 4D data or 3D data with ≥3 components."
            )

    def to_image_layer(self, cmap: str = "gray", alpha: float = 1.0) -> ImageLayer:
        """Create an image layer from the current slice."""
        if self.current is None:
            raise ValueError("No slice data available. Call load_slices first.")
            
        slice_data = self.current

        # Handle 4D data (vector field over time)
        if len(slice_data.shape) == 4:
            time_point = 0
            if slice_data.shape[3] > 1:
                time_point = 1
            vector_data = slice_data[..., time_point]
            bg_data = np.sqrt(np.sum(vector_data**2, axis=-1))
            return ImageLayer(np.rot90(bg_data, 3), cmap=cmap, alpha=alpha)

        # Handle 3D data with vector components
        elif len(slice_data.shape) == 3 and slice_data.shape[2] >= 3:
            if slice_data.shape[2] > 3:
                bg_data = slice_data[..., 0]
            else:
                bg_data = np.sqrt(np.sum(slice_data[..., :3] ** 2, axis=-1))
            return ImageLayer(np.rot90(bg_data, 3), cmap=cmap, alpha=alpha)

        # Handle 3D volumetric data over time
        elif len(slice_data.shape) == 3:
            return ImageLayer(np.rot90(slice_data[..., 0], 3), cmap=cmap, alpha=alpha)

        # Handle 2D data (standard image)
        else:
            return ImageLayer(np.rot90(slice_data, 3), cmap=cmap, alpha=alpha)

    def to_mask_layer(
        self,
        color: str = "r",
        linewidth: float = 1.5,
        smoothing: float = 2.0,
        threshold: float = 0.1,
    ) -> MaskContourLayer:
        if self.current is None:
            raise ValueError("No slice data available. Call load_slices first.")

        mask_data = self.current
        return MaskContourLayer(
            np.rot90(mask_data.astype(np.float32), 3),
            color=color,
            linewidth=linewidth,
            smoothing=smoothing,
            threshold=threshold,
        )

class CroppedSlice(Slice):
    def __init__(self, slice: "Slice", padding: int = 0):
        super().__init__(slice.slice_indices, slice.data)
        self.padding = padding
        self.crop_bounds: Dict[str, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
            "sagittal": None,
            "coronal": None,
            "axial": None
        }
        # Calculate crop bounds for all planes
        self._calculate_all_crop_bounds()

    def _calculate_all_crop_bounds(self) -> None:
        """Calculate crop bounds for all three planes."""
        if self.sagittal is None or self.coronal is None or self.axial is None:
            raise ValueError("Slices must be loaded before cropping.")
        for view_type in ["sagittal", "coronal", "axial"]:
            self.crop_bounds[view_type] = self.find_single_crop_bounds(self[view_type], self.padding)

    def crop_to_nonzero(self) -> None:
        """Crop all planes to their respective non-zero regions."""
        if self.sagittal is None or self.coronal is None or self.axial is None:
            raise ValueError("Slices must be loaded before cropping.")

        # Crop each plane with its own bounds
        for view_type in ["sagittal", "coronal", "axial"]:
            if self.crop_bounds[view_type]:
                self[view_type] = self._apply_single_crop_bounds(self[view_type], self.crop_bounds[view_type])

    def _apply_single_crop_bounds(self, slice_data: np.ndarray, bounds: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        rmin, rmax = bounds[0]
        cmin, cmax = bounds[1]

        has_extra_dims = slice_data.ndim > 2
        
        if has_extra_dims:
            return slice_data[rmin:rmax+1, cmin:cmax+1, ...]
        else:
            return slice_data[rmin:rmax+1, cmin:cmax+1]

    def apply_crop_bounds(self, crop_bounds: Dict[str, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]) -> None:
        if self.sagittal is None or self.coronal is None or self.axial is None:
            raise ValueError("Slices must be loaded before cropping.")
        
        if "sagittal" in crop_bounds and crop_bounds["sagittal"] is not None and self.sagittal is not None:
            self.sagittal = self._apply_single_crop_bounds(self.sagittal, crop_bounds["sagittal"])
            
        if "coronal" in crop_bounds and crop_bounds["coronal"] is not None and self.coronal is not None:
            self.coronal = self._apply_single_crop_bounds(self.coronal, crop_bounds["coronal"])
            
        if "axial" in crop_bounds and crop_bounds["axial"] is not None and self.axial is not None:
            self.axial = self._apply_single_crop_bounds(self.axial, crop_bounds["axial"])
        
    def get_crop_bounds(self, view_type: Optional[str] = None) -> Dict[str, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Get crop bounds for all planes or a specific plane."""
        if view_type:
            if view_type not in self.crop_bounds:
                raise ValueError(f"Invalid view type: {view_type}. Must be sagittal, coronal, or axial.")
            return {view_type: self.crop_bounds[view_type]}
        return self.crop_bounds