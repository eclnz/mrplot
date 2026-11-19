"""Colour assignments shared by categorical fills and outline overlays."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mrplot.colors import CATEGORICAL_PALETTES, DEFAULT_CATEGORICAL_SCHEME, RGB, CategoricalPalette

CATEGORICAL_SCHEMES = tuple(CATEGORICAL_PALETTES)


def _parse_hex(value: str) -> RGB:
    if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise ValueError(f"Invalid hex colour {value!r}. Expected #RRGGBB.")
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def _lookup_hex(fields: list[str]) -> str:
    """Read hex, FreeSurfer (name then byte RGBA), or FSLeyes (unit RGB then name)."""
    if len(fields) == 1:
        _parse_hex(fields[0])
        return fields[0]
    if len(fields) >= 5 and all(re.fullmatch(r"[0-9]+", field) for field in fields[-4:]):
        red, green, blue, alpha = (int(field) for field in fields[-4:])
        if max(red, green, blue, alpha) > 255:
            raise ValueError("FreeSurfer LUT channels must be between 0 and 255.")
        return f"#{red:02X}{green:02X}{blue:02X}"
    if len(fields) >= 4:
        try:
            rgb = [float(field) for field in fields[:3]]
        except ValueError as error:
            raise ValueError("Expected a FreeSurfer or FSLeyes LUT row.") from error
        if not all(0 <= channel <= 1 for channel in rgb):
            raise ValueError("FSLeyes LUT channels must be between 0 and 1.")
        return "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)
    raise ValueError("Expected a hex colour, a FreeSurfer LUT row, or an FSLeyes LUT row.")


def _unused_rgb(color: RGB, seen: set[RGB]) -> RGB:
    integer = (color[0] << 16) | (color[1] << 8) | color[2]
    while color in seen:
        integer = (integer + 1) % (1 << 24)
        color = integer >> 16, (integer >> 8) & 255, integer & 255
    return color


def _scheme_colors(palette: CategoricalPalette, count: int) -> list[RGB]:
    if count > (1 << 24):
        raise ValueError("There are more foreground labels than distinct RGB colours.")
    colors = list(palette.initial[:count])
    seen = set(colors)
    for index, (saturation, value) in zip(range(count - len(colors)), cycle(palette.saturation_values)):
        hue = (index * palette.hue_step) % 1.0
        color = cast(RGB, tuple(round(channel * 255) for channel in hsv_to_rgb(hue, saturation, value)))
        color = _unused_rgb(color, seen)
        colors.append(color)
        seen.add(color)
    return colors


def _image_labels(data: NDArray[Any], name: str) -> set[int]:
    if not data.size or data.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain nonempty, finite integer voxel labels.")
    values = np.unique(data)
    if data.dtype.kind == "f" and (
        not np.all(np.isfinite(values)) or not np.all(values == np.trunc(values))
    ):
        raise ValueError(f"{name} must contain finite integer voxel labels.")
    return {int(value) for value in values}


def _validate_mapping_coverage(labels: list[int], mapping: Mapping[int, str], name: str) -> None:
    missing = [label for label in labels if label not in mapping]
    if missing:
        shown = ", ".join(map(str, missing[:10]))
        suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(f"{name} colour mapping is missing labels {shown}{suffix}.")


@dataclass(frozen=True, slots=True)
class CategoricalColors:
    """Assign colours to foreground labels in ascending numeric order or by explicit key."""

    scheme: str = DEFAULT_CATEGORICAL_SCHEME
    background_label: int = 0
    mapping: Mapping[int, str] | None = None
    colors: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.scheme not in CATEGORICAL_SCHEMES:
            raise ValueError(f"Unknown categorical colour scheme {self.scheme!r}.")
        if not isinstance(self.background_label, (int, np.integer)):
            raise ValueError("The background label must be an integer.")
        if self.mapping is not None and self.colors is not None:
            raise ValueError("Supply either a label-to-colour mapping or a positional colour list.")
        if self.mapping is not None and any(not isinstance(label, (int, np.integer)) for label in self.mapping):
            raise ValueError("Colour mapping keys must be integer labels.")
        if isinstance(self.colors, str):
            raise ValueError("Supply outline colours as a list, even for a single foreground region.")
        if self.mapping is not None:
            object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))
        if self.colors is not None:
            object.__setattr__(self, "colors", tuple(self.colors))

    @classmethod
    def from_lookup_table(
        cls, path: str | os.PathLike[str], *, background_label: int = 0,
    ) -> CategoricalColors:
        """Read a LABEL #RRGGBB, FreeSurfer, or FSLeyes lookup table."""
        mapping = {}
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            label, *fields = line.split()
            if re.fullmatch(r"[+-]?[0-9]+", label) is None:
                raise ValueError(f"{path}, line {number}: expected an integer label.")
            key = int(label)
            if key in mapping:
                raise ValueError(f"{path}, line {number}: duplicate label {key}.")
            try:
                mapping[key] = _lookup_hex(fields)
            except ValueError as error:
                raise ValueError(f"{path}, line {number}: {error}") from error
        return cls(background_label=background_label, mapping=mapping)

    @classmethod
    def from_color_file(
        cls, path: str | os.PathLike[str], *, background_label: int = 0,
    ) -> CategoricalColors:
        """Read one hex colour per line in ascending foreground label order."""
        colors = tuple(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines())
        return cls(background_label=background_label, colors=colors)

    def resolve(
        self, *volumes: NDArray[Any], name: str = "Parcellation", max_values: int | None = None,
    ) -> LabelColors:
        labels = set().union(*(_image_labels(data, name) for data in volumes))
        if max_values is not None and len(labels) > max_values:
            raise ValueError(
                f"{name} has {len(labels)} unique voxel values. The limit is {max_values}, including background."
            )
        labels.discard(self.background_label)
        foreground = sorted(labels)
        colors = self._resolve_colors(foreground, name)
        indices = {label: index for index, label in enumerate(foreground, start=1)}
        indices[self.background_label] = 0
        return LabelColors(indices, tuple(colors))

    def _resolve_colors(self, foreground: list[int], name: str) -> list[RGB]:
        if self.mapping is not None:
            _validate_mapping_coverage(foreground, self.mapping, name)
            colors_by_label = {label: _parse_hex(color) for label, color in self.mapping.items()}
            return [colors_by_label[label] for label in foreground]
        if self.colors is None:
            return _scheme_colors(CATEGORICAL_PALETTES[self.scheme], len(foreground))
        if len(self.colors) != len(foreground):
            raise ValueError(
                f"{name} expects {len(foreground)} foreground region colours, got {len(self.colors)}."
            )
        return [_parse_hex(color) for color in self.colors]


@dataclass(frozen=True, slots=True)
class LabelColors:
    """Resolved colour table, with index zero reserved for background."""

    indices: Mapping[int, int]
    foreground: tuple[RGB, ...]

    def encode(self, data: NDArray[Any]) -> NDArray[np.int32]:
        labels, inverse = np.unique(data, return_inverse=True)
        indices = np.array([self.indices[int(label)] for label in labels], dtype=np.int32)
        return cast(NDArray[np.int32], indices[inverse].reshape(data.shape))

    def palette(self, background: RGB) -> NDArray[np.uint8]:
        return np.array((background, *self.foreground), dtype=np.uint8)
