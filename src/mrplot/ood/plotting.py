import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, TypeVar, Union
from skimage import measure
from scipy.ndimage import gaussian_filter  # type: ignore
import abc

# Define type variables for return types
T = TypeVar("T")


class PlotLayer(abc.ABC):
    """Abstract base class for all plot layers"""

    @abc.abstractmethod
    def render(self, ax: plt.Axes) -> Any:
        """Render the layer on the given axes"""
        pass


class VectorFieldLayer(PlotLayer):
    def __init__(
        self,
        vector: np.ndarray,
        slice_type: str = "sagittal",
        thin_factor: int = 6,
        scale: float = 10.0,
        cmap: str = "viridis",
        min_magnitude: float = 1e-6,  # Threshold for considering arrow length as zero
        **quiver_kwargs: Any,
    ) -> None:

        self.vector = vector
        self.slice_type = slice_type
        self.thin_factor = thin_factor
        self.scale = scale
        self.cmap = cmap
        self.min_magnitude = min_magnitude
        self.quiver_kwargs = quiver_kwargs

        # Default quiver settings
        self.default_quiver_kwargs: Dict[str, Any] = {
            "scale": 25,
            "scale_units": "width",
            "width": 0.002,
            "headwidth": 6,
            "headlength": 7,
            "headaxislength": 6,
            "pivot": "tail",
        }

        # Update defaults with provided kwargs
        self.default_quiver_kwargs.update(quiver_kwargs)

    def render(self, ax: plt.Axes) -> Any:
        x_dim, y_dim, component_dim = self.vector.shape

        if component_dim != 3:
            raise ValueError(f"Expected 3 components (x,y,z) but got {component_dim}")

        # Create a grid of coordinates
        xx, yy = np.meshgrid(np.arange(x_dim), np.arange(y_dim), indexing="ij")

        # Filter coordinates based on thin_factor
        mask_arrow = (xx % self.thin_factor == 0) & (yy % self.thin_factor == 0)
        xx_masked = xx[mask_arrow]
        yy_masked = yy[mask_arrow]

        # Extract displacement components based on slice type
        if self.slice_type == "axial":
            u = self.vector[xx_masked, yy_masked, 0] * self.scale
            v = self.vector[xx_masked, yy_masked, 1] * self.scale
        elif self.slice_type == "coronal":
            u = self.vector[xx_masked, yy_masked, 0] * self.scale
            v = self.vector[xx_masked, yy_masked, 2] * self.scale
        elif self.slice_type == "sagittal":
            u = self.vector[xx_masked, yy_masked, 1] * self.scale
            v = self.vector[xx_masked, yy_masked, 2] * self.scale
        else:
            raise ValueError(
                "Invalid slice type. Must be 'axial', 'coronal', or 'sagittal'."
            )

        # Calculate total displacement magnitude
        total_displacement = np.sqrt(u**2 + v**2)

        # Create alpha values based on displacement magnitude
        # Set alpha to 0 for very small displacements
        alphas = np.ones_like(total_displacement)
        alphas[total_displacement < self.min_magnitude] = 0.0

        # Plot vectors using quiver
        quiver = ax.quiver(
            xx_masked,
            yy_masked,
            u,
            v,
            total_displacement,
            cmap=self.cmap,
            alpha=alphas,  # Use our calculated alpha values
            **self.default_quiver_kwargs,
        )

        return quiver


class MaskContourLayer(PlotLayer):

    def __init__(
        self,
        mask: np.ndarray,
        color: str = "r",
        linewidth: float = 1.0,
        smoothing: float = 2.0,
        threshold: float = 0.1,
    ) -> None:

        self.mask = mask.astype(np.float32)  # Convert to float32 for gaussian filter
        self.color = color
        self.linewidth = linewidth
        self.smoothing = smoothing
        self.threshold = threshold

    def render(self, ax: plt.Axes) -> List[plt.Line2D]:

        # Smooth the mask with Gaussian filter
        smoothed_mask = gaussian_filter(self.mask, sigma=self.smoothing)

        # Find the contours of the smoothed mask
        contours = measure.find_contours(smoothed_mask, self.threshold)

        # Store the contour line objects
        contour_lines: List[plt.Line2D] = []

        # Plot each contour
        for contour in contours:
            # Only plot contours with reasonable number of points
            if len(contour) > 5:
                # Subsample large contours for efficiency
                if len(contour) > 1000:
                    contour = contour[:: len(contour) // 500]

                # Plot the contour
                (line,) = ax.plot(
                    contour[:, 1],
                    contour[:, 0],
                    color=self.color,
                    linewidth=self.linewidth,
                )
                contour_lines.append(line)

        return contour_lines


class ImageLayer(PlotLayer):

    def __init__(self, data: np.ndarray, cmap: str = "gray", alpha: float = 1.0):
        self.data = data
        self.cmap = cmap
        self.alpha = alpha

    def render(self, ax: plt.Axes) -> None:
        ax.imshow(self.data, cmap=self.cmap, alpha=self.alpha, origin="lower")


class PlotComposer:
    """Class for composing multiple plot layers in a grid layout"""

    def __init__(self, figsize: Tuple[int, int] = (10, 10), title: str = "") -> None:
        self.layers: List[PlotLayer] = []
        self.figsize = figsize
        self.title = title
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.subplots: Dict[int, List[PlotLayer]] = {}
        self.current_subplot: int = 0
        self.rows: int = 1
        self.cols: int = 1
        self._subplot_labels: Dict[int, Dict[str, str]] = {}
        self._subplot_ticks: Dict[int, Dict[str, Any]] = {}
        self._global_tick_settings: Dict[str, Any] = {
            "show_ticks": True,
            "show_tick_labels": True,
            "tick_size": 10,
        }
        self.title_fontsize: int = 14

    def add_subplot(self, rows: int, cols: int, index: int) -> "PlotComposer":
        if index > rows * cols:
            raise ValueError(f"Index {index} exceeds subplot grid size ({rows}x{cols})")
        self.rows = rows
        self.cols = cols
        self.current_subplot = index - 1  # Convert to 0-based index
        if self.current_subplot not in self.subplots:
            self.subplots[self.current_subplot] = []
        return self

    def add_layer(self, layer: PlotLayer) -> "PlotComposer":
        """Add a layer to either the main plot or current subplot."""
        if self.current_subplot == 0 and not self.subplots:
            # Adding to main plot
            self.layers.append(layer)
        else:
            # Adding to subplot
            self.subplots[self.current_subplot].append(layer)
        return self

    def set_labels(
        self,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        subplot_idx: Optional[int] = None,
    ) -> "PlotComposer":

        if subplot_idx is not None:
            # Convert to 0-based index for internal use
            idx = subplot_idx - 1
        else:
            idx = self.current_subplot

        # Store the labels to apply during rendering
        if idx not in self._subplot_labels:
            self._subplot_labels[idx] = {}

        if xlabel is not None:
            self._subplot_labels[idx]["xlabel"] = xlabel
        if ylabel is not None:
            self._subplot_labels[idx]["ylabel"] = ylabel
        if title is not None:
            self._subplot_labels[idx]["title"] = title

        return self

    def set_figure_title(self, title: str, fontsize: int = 14) -> "PlotComposer":
        self.title = title
        self.title_fontsize = fontsize
        return self

    def set_ticks(
        self,
        show_ticks: bool = True,
        show_tick_labels: bool = True,
        x_ticks: Optional[List] = None,
        y_ticks: Optional[List] = None,
        x_ticklabels: Optional[List[str]] = None,
        y_ticklabels: Optional[List[str]] = None,
        tick_size: int = 10,
        subplot_idx: Optional[int] = None,
    ) -> "PlotComposer":
        # Convert to 0-based indexing internally
        subplot_idx_internal = (
            self.current_subplot if subplot_idx is None else subplot_idx - 1
        )

        # Initialize tick settings dictionary if it doesn't exist
        if not hasattr(self, "_subplot_ticks"):
            self._subplot_ticks = {}

        if subplot_idx_internal not in self._subplot_ticks:
            self._subplot_ticks[subplot_idx_internal] = {}

        # Store tick configuration
        tick_config = self._subplot_ticks[subplot_idx_internal]
        tick_config["show_ticks"] = show_ticks
        tick_config["show_tick_labels"] = show_tick_labels
        tick_config["x_ticks"] = x_ticks
        tick_config["y_ticks"] = y_ticks
        tick_config["x_ticklabels"] = x_ticklabels
        tick_config["y_ticklabels"] = y_ticklabels
        tick_config["tick_size"] = tick_size

        return self

    def set_all_ticks(
        self,
        show_ticks: bool = False,
        show_tick_labels: bool = False,
        tick_size: int = 10,
    ) -> "PlotComposer":
        if not hasattr(self, "_global_tick_settings"):
            self._global_tick_settings = {}

        self._global_tick_settings["show_ticks"] = show_ticks
        self._global_tick_settings["show_tick_labels"] = show_tick_labels
        self._global_tick_settings["tick_size"] = tick_size

        return self

    def render(self) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        if self.subplots:
            # Create subplot grid
            fig, axes = plt.subplots(self.rows, self.cols, figsize=self.figsize)
            self.fig = fig

            if self.title:
                fig.suptitle(self.title, fontsize=self.title_fontsize)

            # Handle both single and multiple subplots
            if self.rows * self.cols == 1:
                axes = np.array([axes])
            axes_flat = axes.flatten()

            # Render each subplot
            for idx, layers in self.subplots.items():
                if idx < len(axes_flat):
                    ax = axes_flat[idx]
                    for layer in layers:
                        layer.render(ax)
                    ax.set_aspect("equal")

                    # Apply any labels that were set
                    if idx in self._subplot_labels:
                        labels = self._subplot_labels[idx]
                        if "xlabel" in labels:
                            ax.set_xlabel(labels["xlabel"])
                        if "ylabel" in labels:
                            ax.set_ylabel(labels["ylabel"])
                        if "title" in labels:
                            ax.set_title(labels["title"])

                    # Apply tick settings
                    # First check global settings
                    show_ticks = self._global_tick_settings.get("show_ticks", True)
                    show_tick_labels = self._global_tick_settings.get(
                        "show_tick_labels", True
                    )
                    tick_size = self._global_tick_settings.get("tick_size", 10)

                    # Override with subplot-specific settings if they exist
                    if hasattr(self, "_subplot_ticks") and idx in self._subplot_ticks:
                        tick_config = self._subplot_ticks[idx]
                        show_ticks = tick_config.get("show_ticks", show_ticks)
                        show_tick_labels = tick_config.get(
                            "show_tick_labels", show_tick_labels
                        )
                        tick_size = tick_config.get("tick_size", tick_size)

                        # Apply custom tick positions if provided
                        if tick_config.get("x_ticks") is not None:
                            ax.set_xticks(tick_config["x_ticks"])
                        if tick_config.get("y_ticks") is not None:
                            ax.set_yticks(tick_config["y_ticks"])

                        # Apply custom tick labels if provided
                        if tick_config.get("x_ticklabels") is not None:
                            ax.set_xticklabels(tick_config["x_ticklabels"])
                        if tick_config.get("y_ticklabels") is not None:
                            ax.set_yticklabels(tick_config["y_ticklabels"])

                    # Set tick visibility
                    if not show_ticks:
                        ax.tick_params(axis="both", which="both", length=0)

                    # Set tick label visibility
                    if not show_tick_labels:
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])

                    # Set tick parameters
                    ax.tick_params(axis="both", which="major", labelsize=tick_size)

            return fig, axes

        else:
            # Original single plot rendering
            fig, ax = plt.subplots(figsize=self.figsize)
            self.fig, self.ax = fig, ax

            for layer in self.layers:
                layer.render(ax)

            if self.title:
                fig.suptitle(self.title, fontsize=self.title_fontsize)

            ax.set_aspect("equal")
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.tick_params(axis="both", which="minor", labelsize=8)

            return fig, ax

    def show(self) -> None:
        """Display the plot(s)."""
        if self.fig is None:
            self.render()
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str, dpi: int = 300) -> None:
        """Save the plot(s) to a file."""
        if self.fig is None:
            self.render()
        assert self.fig is not None  # For type checking
        self.fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
