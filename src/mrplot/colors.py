"""Built-in continuous color scales and categorical palette definitions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ColorScale:
    """Evenly spaced RGB stops spanning the intensity window from low to high."""

    stops: tuple[RGB, ...]
    palette: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.stops) < 2:
            raise ValueError("A color scale requires at least two RGB stops.")
        maximum = np.iinfo(np.uint8).max
        for color in self.stops:
            if len(color) != 3 or any(
                not isinstance(channel, (int, np.integer)) or channel < 0 or channel > maximum
                for channel in color
            ):
                raise ValueError("Color scale stops must contain three uint8 RGB channels.")
        stops = np.asarray(self.stops)
        positions = np.linspace(0, maximum, len(self.stops))
        levels = np.arange(maximum + 1)
        channels = [np.interp(levels, positions, channel) for channel in stops.T]
        palette = np.rint(np.column_stack(channels)).astype(np.uint8)
        object.__setattr__(self, "palette", palette.tobytes())


GRAY = ColorScale(((0, 0, 0), (255, 255, 255)))
HEAT = ColorScale(((0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)))
BLUE_RED = ColorScale(((0, 0, 255), (255, 255, 255), (255, 0, 0)))

COLOR_SCALES = {"gray": GRAY, "heat": HEAT, "blue-red": BLUE_RED}


@dataclass(frozen=True, slots=True)
class CategoricalPalette:
    """Initial RGB colours and HSV parameters for extending the palette."""

    initial: tuple[RGB, ...]
    hue_step: float
    saturation_values: tuple[tuple[float, float], ...]


_VIVID_PALETTE = CategoricalPalette(
    initial=(),
    hue_step=0.618033988749895,
    saturation_values=((0.85, 0.95), (0.65, 0.78), (0.95, 0.68)),
)
_TABLEAU_PALETTE = replace(_VIVID_PALETTE, initial=(
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
))

DEFAULT_CATEGORICAL_SCHEME = "tableau"
CATEGORICAL_PALETTES = MappingProxyType({
    DEFAULT_CATEGORICAL_SCHEME: _TABLEAU_PALETTE,
    "vivid": _VIVID_PALETTE,
})
