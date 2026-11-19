"""Render synthetic NIfTI volumes through PNG and MP4 encoding."""

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from mrplot.categorical import CategoricalColors
from mrplot.overlays import LabelOverlay
from mrplot.plotting import render_file
from mrplot.rendering import RenderConfig
from tests.phantoms import box_mask, ellipsoid, moving_sphere, save_phantom


def test_ellipsoid_pngs_preserve_physical_aspect_ratio(tmp_path):
    shape = (24, 32, 40)
    phantom = save_phantom(tmp_path, "ellipsoid", ellipsoid(shape), np.diag([2, 1.5, 1, 1]))
    paths = render_file(phantom, tmp_path / "out", config=RenderConfig(max_fig_size=192))

    # The physical extent is 48 × 48 × 40 mm.
    for path, expected_size in zip(paths, [(192, 160), (192, 160), (192, 192)], strict=True):
        with Image.open(path) as frame:
            assert frame.size == expected_size
            assert frame.getpixel((frame.width // 2, frame.height // 2))[0] > 220
            assert frame.getpixel((4, frame.height - 5)) == (0, 0, 0)


def test_two_region_phantom_preserves_neighbouring_roi_boundaries(tmp_path):
    shape = (24, 24, 24)
    x = np.indices(shape, sparse=True)[0]
    labels = box_mask(shape, half_width=7) * np.where(x < 12, 3, 42).astype(np.uint8)
    anatomy = save_phantom(tmp_path, "ellipsoid", ellipsoid(shape))
    roi = save_phantom(tmp_path, "two_regions", labels)
    colors = CategoricalColors(mapping={3: "#E41A1C", 42: "#377EB8"})

    _, _, axial = render_file(
        anatomy, tmp_path / "out", config=RenderConfig(max_fig_size=192),
        overlays=[LabelOverlay(roi, colors=colors)],
    )
    with Image.open(axial) as frame:
        assert frame.getpixel((95, 96)) == (55, 126, 184)
        assert frame.getpixel((96, 96)) == (228, 26, 28)
        r, g, b = frame.getpixel((100, 96))
        assert r == g == b


def test_moving_sphere_remains_in_order_in_encoded_video(tmp_path):
    phantom = save_phantom(tmp_path, "moving_sphere", moving_sphere())
    paths = render_file(phantom, tmp_path / "out", config=RenderConfig(max_fig_size=192))

    for path in paths:
        with imageio.get_reader(path) as video:
            assert video.count_frames() == 2
    with imageio.get_reader(paths[2]) as axial:
        first, second = axial.get_data(0), axial.get_data(1)

    # Sample inside each sphere's axial position, away from compressed edges and text.
    left = np.s_[84:104, 48:66, :]
    right = np.s_[84:104, 126:144, :]
    assert first[right].mean() > 220
    assert first[left].mean() < 20
    assert second[left].mean() > 220
    assert second[right].mean() < 20
