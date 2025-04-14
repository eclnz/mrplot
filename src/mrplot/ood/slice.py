import nibabel as nib
from typing import Optional, Tuple, List, Dict, Any, Set, Union
import numpy as np
import re
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
        subject_id: str,
        session_id: str,
        scan_id: str,
        path: Optional[str] = None,
        slice_indices: Optional[Tuple[int, int, int]] = None,
        origin: Optional[str] = None,
        data: Optional[np.ndarray] = None,
    ):
        self.subject_id = subject_id
        self.session_id = session_id
        self.scan_id = scan_id
        self.path = path
        self.slice_indices = slice_indices
        self.sagittal, self.coronal, self.axial = self._load_slices(slice_indices, data)
        self.current = self.sagittal
        self.current_view = "sagittal"
        self.origin = origin

    def _load_slices(
        self,
        slice_indices: Optional[Tuple[int, int, int]] = None,
        data: Optional[np.ndarray] = None,
    ):
        if self.path is not None:
            img = nib.as_closest_canonical(nib.load(self.path))
            volume = np.asarray(img.dataobj).astype(np.float32)
        elif data is not None:
            volume = np.asarray(data).astype(np.float32)
        else:
            raise ValueError("Either path or data must be provided.")
        
        if slice_indices is None:
            slice_indices = (
                volume.shape[0] // 2,
                volume.shape[1] // 2,
                volume.shape[2] // 2,
            )

        sagittal = volume[slice_indices[0], :, :, ...]
        coronal = volume[:, slice_indices[1], :, ...]
        axial = volume[:, :, slice_indices[2], ...]
        
        del volume, img
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
        
        mask_data = self.current
        return MaskContourLayer(
            np.rot90(mask_data.astype(np.float32), 3),
            color=color,
            linewidth=linewidth,
            smoothing=smoothing,
            threshold=threshold,
        )


class SliceCollection:
    def __init__(self, slices: Optional[List[Slice]] = None):
        self.slices: List[Slice] = slices if slices is not None else []
        # Store actual slice objects instead of indices
        self.groups: Dict[str, Dict[SliceType, Set[Slice]]] = {}
        # Track relationships between slices
        self.relations: Dict[Slice, Dict[str, Set[Slice]]] = {}
        
    def __iter__(self):
        return iter(self.slices)
    
    def add_slice(self, slice_obj: Slice, slice_type: SliceType) -> None:
        self.slices.append(slice_obj)
        
        # Group by subject/session as base key
        group_key = f"{slice_obj.subject_id}/{slice_obj.session_id}"
        if group_key not in self.groups:
            self.groups[group_key] = {}
        if slice_type not in self.groups[group_key]:
            self.groups[group_key][slice_type] = set()
            
        self.groups[group_key][slice_type].add(slice_obj)

    def add_slices(self, slices: Union[List[Slice], "SliceCollection"], slice_type: SliceType) -> None:
        if isinstance(slices, SliceCollection):
            slices = slices.slices
        
        # Track which groups we're adding to
        affected_groups: Dict[str, int] = {}
        
        for slice_obj in slices:
            group_key = f"{slice_obj.subject_id}/{slice_obj.session_id}"
            affected_groups[group_key] = affected_groups.get(group_key, 0) + 1
            
            # Check if this type already exists in the group
            if (group_key in self.groups and 
                slice_type in self.groups[group_key] and 
                len(self.groups[group_key][slice_type]) > 0):
                print(f"Warning: Group {group_key} already has slices of type {slice_type.name}")
            
            self.add_slice(slice_obj, slice_type)
        
        # Check for uneven groups
        for group_key, type_dict in self.groups.items():
            sizes = {slice_type.name: len(slices) for slice_type, slices in type_dict.items()}
            if len(set(sizes.values())) > 1:
                print(f"Warning: Group {group_key} has uneven numbers of slices:")
                for type_name, size in sizes.items():
                    print(f"  - {type_name}: {size} slice(s)")
                    
        # Check if any affected groups have different numbers of slices
        affected_sizes = {group: count for group, count in affected_groups.items()}
        if len(set(affected_sizes.values())) > 1:
            print(f"Warning: Newly added slices have uneven distribution across groups:")
            for group, count in affected_sizes.items():
                print(f"  - {group}: {count} slice(s) of type {slice_type.name}")

    def relate_slices(self, source: Slice, target: Slice, relation: SliceRelation) -> None:
        if source not in self.relations:
            self.relations[source] = {}
        if relation.relation_name not in self.relations[source]:
            self.relations[source][relation.relation_name] = set()
            
        self.relations[source][relation.relation_name].add(target)

    def get_related(self, slice_obj: Slice, relation_name: str) -> Set[Slice]:
        return self.relations.get(slice_obj, {}).get(relation_name, set())

    def find_slices(self, 
                   subject_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   slice_type: Optional[SliceType] = None) -> Set[Slice]:
        results = set()
        for group_key, type_dict in self.groups.items():
            sub, ses = group_key.split('/')
            if (subject_id and sub != subject_id) or (session_id and ses != session_id):
                continue
            if slice_type:
                results.update(type_dict.get(slice_type, set()))
            else:
                for slices in type_dict.values():
                    results.update(slices)
        return results

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> Slice:
        if not 0 <= index < len(self.slices):
            raise IndexError("Slice index out of range.")
        return self.slices[index]

    def filter(self, criteria: Dict[str, Any]) -> "SliceCollection":
        valid_attributes = {"subject_id", "session_id", "scan_id", "path", "slice_indices", "current_view"}
        invalid_keys = set(criteria.keys()) - valid_attributes
        if invalid_keys:
            raise ValueError(f"Invalid filter criteria: {invalid_keys}. Valid attributes are: {valid_attributes}")
        
        filtered_slices = []
        for slice_obj in self.slices:
            matches = True
            for key, value in criteria.items():
                try:
                    attr_value = getattr(slice_obj, key)
                    # Handle regex patterns for string attributes
                    if isinstance(value, str) and isinstance(attr_value, str):
                        if not re.match(value, attr_value):
                            matches = False
                            break
                    # Handle exact matches for non-string attributes
                    elif attr_value != value:
                        matches = False
                        break
                except AttributeError as e:
                    raise AttributeError(f"Error accessing attribute '{key}' on slice: {e}")
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{value}' for {key}: {e}")
            
            if matches:
                filtered_slices.append(slice_obj)
        
        if not filtered_slices:
            print(f"Warning: No slices match the filter criteria: {criteria}")
        
        return SliceCollection()

    def set_view_all(self, view_type: str) -> None:
        """Sets the view for all slices in the collection."""
        for s in self.slices:
            s.set_view(view_type)
            
    def plot(
        self, 
        view_type: str = "sagittal",
        plot_kwargs: Optional[Dict[SliceType, Dict[str, Any]]] = None,
        title: Optional[str] = None
    ) -> None:
        # Set default plotting parameters if none provided
        default_kwargs: Dict[SliceType, Dict[str, Any]] = {
            SliceType.MOTION: {'thin_factor': 6, 'scale': 10.0},
            SliceType.T1: {'cmap': 'gray', 'alpha': 1.0},
            SliceType.T2: {'cmap': 'gray', 'alpha': 1.0},
            SliceType.MASK: {'color': 'r', 'linewidth': 1.5},
            SliceType.BOLD: {'cmap': 'gray', 'alpha': 1.0},
            SliceType.FIELDMAP: {'cmap': 'gray', 'alpha': 1.0}
        }
        if title is None:
            title = f"Slice Collection - {view_type} view"
        composer = PlotComposer(title=title)
        
        # Update defaults with user-provided kwargs
        if plot_kwargs:
            for slice_type, kwargs in plot_kwargs.items():
                if slice_type in default_kwargs:
                    default_kwargs[slice_type] = {**default_kwargs[slice_type], **kwargs}
        
        # Count number of groups
        num_groups = len(self.groups)
        if num_groups == 0:
            return
            
        # Calculate grid dimensions
        cols = min(3, num_groups)  # Maximum 3 columns
        rows = (num_groups + cols - 1) // cols  # Ceiling division
        
        # Plot each group in its own subplot
        for idx, (group_key, type_dict) in enumerate(self.groups.items(), 1):
            # Create new subplot for this group
            composer.add_subplot(rows, cols, idx)
            
            # Set view type for all slices in this group
            for slices in type_dict.values():
                for slice_obj in slices:
                    slice_obj.set_view(view_type)
            
            # Add layers for each slice type in the group
            for slice_type, slices in type_dict.items():
                plot_settings = dict(default_kwargs.get(slice_type, {}))
                for slice_obj in slices:
                    if slice_type in [SliceType.T1, SliceType.T2, SliceType.BOLD, SliceType.FIELDMAP]:
                        composer.add_layer(slice_obj.to_image_layer(**plot_settings))
                    elif slice_type == SliceType.MOTION:
                        composer.add_layer(slice_obj.to_vector_layer(**plot_settings))
                    elif slice_type == SliceType.MASK:
                        composer.add_layer(slice_obj.to_mask_layer(**plot_settings))
        composer.show()

    def print_tree(self, include_details: bool = False) -> None:
        """Print a visual tree representation of the slice collection structure.
        
        Args:
            include_details: If True, include additional details like paths and indices
        """
        if not self.slices:
            print("Empty collection")
            return
            
        nodes = []
        
        if not self.groups:
            # Untyped collection (from create_slices) - group by subject/session
            groups: Dict[str, Set[Slice]] = {}
            for slice_obj in self.slices:
                key = f"{slice_obj.subject_id}/{slice_obj.session_id}"
                if key not in groups:
                    groups[key] = set()
                groups[key].add(slice_obj)
            
            # Create the tree structure
            for group_key, slices in groups.items():
                nodes.append((0, f"Group: {group_key}"))
                
                for i, slice_obj in enumerate(sorted(slices, key=lambda x: x.scan_id)):
                    details = f"Slice: {slice_obj.scan_id}"
                    if include_details:
                        if slice_obj.path:
                            details += f", path: {slice_obj.path}"
                        if slice_obj.slice_indices:
                            details += f", indices: {slice_obj.slice_indices}"
                    nodes.append((1, details))
        else:
            # Typed collection (after add_slices)
            for group_key, type_dict in self.groups.items():
                nodes.append((0, f"Group: {group_key}"))
                
                for slice_type, slices in type_dict.items():
                    nodes.append((1, f"{slice_type.name} ({len(slices)} slices)"))
                    
                    if slices:  # Only proceed if there are slices of this type
                        for slice_obj in sorted(list(slices), key=lambda x: x.scan_id):
                            details = f"Slice: {slice_obj.scan_id}"
                            if include_details:
                                if slice_obj.path:
                                    details += f", path: {slice_obj.path}"
                                if slice_obj.slice_indices:
                                    details += f", indices: {slice_obj.slice_indices}"
                            nodes.append((2, details))
        
        print_tree(nodes)
