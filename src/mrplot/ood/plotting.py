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

    def render(self) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        if self.subplots:
            # Create subplot grid
            fig, axes = plt.subplots(self.rows, self.cols, figsize=self.figsize)
            self.fig = fig

            if self.title:
                fig.suptitle(self.title)

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
                    ax.tick_params(axis="both", which="major", labelsize=10)
                    ax.tick_params(axis="both", which="minor", labelsize=8)

            return fig, axes

        else:
            # Original single plot rendering
            fig, ax = plt.subplots(figsize=self.figsize)
            self.fig, self.ax = fig, ax

            for layer in self.layers:
                layer.render(ax)

            if self.title:
                ax.set_title(self.title)

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
