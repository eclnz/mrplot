"""Immutable RGB color scales with lookup tables prepared at construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class ColorScale:
    """Evenly spaced RGB stops spanning the intensity window from low to high."""

    stops: tuple[tuple[int, int, int], ...]
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
