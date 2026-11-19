import numpy as np
import pytest

from mrplot.colors import COLOR_SCALES, ColorScale
from mrplot.rendering import (
    FrameLayout,
    Interpolation,
    RenderConfig,
    calculate_output_size,
    load_render_resources,
    render_image_frame,
    render_prepared_frame,
)


def test_render_image_frame_produces_rgb_uint8_array():
    config = RenderConfig()
    slice_2d = np.linspace(0, 1, 400).reshape(20, 20)
    frame = render_image_frame(slice_2d, (1.0, 1.0), config, (0.0, 1.0), "title")
    assert frame.ndim == 3
    assert frame.shape[-1] == 3
    assert frame.dtype == np.uint8


def test_render_config_rejects_nonpositive_font_size():
    with pytest.raises(ValueError, match="font_size must be positive"):
        RenderConfig(font_size=0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"fps": 0}, "Frames per second must be positive"),
        ({"intensity_max": float("nan")}, "Intensity bounds must be finite"),
    ],
)
def test_render_config_rejects_invalid_output_settings(options, message):
    with pytest.raises(ValueError, match=message):
        RenderConfig(**options)


@pytest.mark.parametrize("show_colorbar", [False, True])
def test_prepared_layout_reuses_resources_without_mutating_prior_frames(show_colorbar):
    first_slice = np.zeros((8, 10))
    second_slice = np.ones((8, 10))
    config = RenderConfig(max_fig_size=64, show_colorbar=show_colorbar)
    layout = FrameLayout(first_slice.shape, (1, 1), (0, 1), load_render_resources(config))
    first = render_prepared_frame(first_slice, layout, "Frame 0", crosshair=(4, 5))
    saved_first = first.copy()
    second = render_prepared_frame(second_slice, layout, "Frame 1", crosshair=(4, 5))

    np.testing.assert_array_equal(first, saved_first)
    np.testing.assert_array_equal(
        second, render_image_frame(second_slice, (1, 1), config, (0, 1), "Frame 1", (4, 5))
    )
    assert not np.array_equal(first, second)
    assert not np.shares_memory(first, second)


def test_calculate_output_size_respects_max_fig_size():
    config = RenderConfig(figure_scale_factor=100, max_fig_size=160)
    width_px, height_px = calculate_output_size((10, 10), (1.0, 1.0), config)
    macro_block_px = 16
    assert width_px <= config.max_fig_size + macro_block_px
    assert height_px <= config.max_fig_size + macro_block_px


def test_calculate_output_size_corrects_anisotropic_voxels():
    """Height:width ratio should track the spacing ratio, not voxel count."""
    config = RenderConfig(figure_scale_factor=10, max_fig_size=100_000)
    square_shape = (20, 20)

    isotropic_width, isotropic_height = calculate_output_size(
        square_shape, (1.0, 1.0), config
    )
    anisotropic_width, anisotropic_height = calculate_output_size(
        square_shape, (2.0, 1.0), config
    )

    assert isotropic_width == isotropic_height
    assert anisotropic_height / anisotropic_width == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("interpolation", list(Interpolation))
def test_render_image_frame_accepts_every_interpolation_mode(interpolation):
    config = RenderConfig(interpolation=interpolation)
    slice_2d = np.linspace(0, 1, 400).reshape(20, 20)
    frame = render_image_frame(slice_2d, (1.0, 1.0), config, (0.0, 1.0), "title")
    assert frame.dtype == np.uint8


def test_nearest_interpolation_keeps_only_source_intensity_values():
    """Nearest-neighbor must not introduce intensity values absent from the source."""
    config = RenderConfig(
        figure_scale_factor=20, max_fig_size=10_000, interpolation=Interpolation.NEAREST
    )
    slice_2d = np.array([[0.0, 1.0], [1.0, 0.0]])
    frame = render_image_frame(slice_2d, (1.0, 1.0), config, (0.0, 1.0), "")
    # Ignore the title text drawn in white - check the bulk of the image only.
    body = frame[20:, :, 0]
    assert set(np.unique(body).tolist()) <= {0, 255}


def test_crosshair_draws_green_pixels():
    config = RenderConfig()
    slice_2d = np.zeros((20, 20))
    plain = render_image_frame(slice_2d, (1.0, 1.0), config, (0.0, 1.0), "")
    with_crosshair = render_image_frame(
        slice_2d, (1.0, 1.0), config, (0.0, 1.0), "", crosshair=(10, 10)
    )
    assert not np.array_equal(plain, with_crosshair)
    assert np.any(np.all(with_crosshair == (0, 255, 0), axis=-1))


@pytest.mark.parametrize(
    "scale",
    [*COLOR_SCALES.values(), ColorScale(((10, 20, 30), (90, 120, 150)))],
)
def test_injected_scale_is_shared_by_image_and_colorbar(scale):
    source = np.array([[0.0, 1.0], [0.0, 1.0]])
    config = RenderConfig(
        max_fig_size=64, interpolation=Interpolation.NEAREST, show_colorbar=True,
        color_scale=scale,
    )
    frame = render_image_frame(source, (1, 1), config, (0, 1), "")
    np.testing.assert_array_equal(frame[32, 16], scale.stops[0])
    np.testing.assert_array_equal(frame[32, 48], scale.stops[-1])
    bar_x = 64 + config.colorbar_margin + config.colorbar_width // 2
    np.testing.assert_array_equal(frame[0, bar_x], scale.stops[-1])
    np.testing.assert_array_equal(frame[-1, bar_x], scale.stops[0])


@pytest.mark.parametrize(
    "stops",
    [((0, 0, 0),), ((-1, 0, 0), (255, 255, 255)), ((0, 0, 0), (256, 255, 255))],
)
def test_color_scale_rejects_invalid_stops_at_construction(stops):
    with pytest.raises(ValueError, match="color scale|Color scale"):
        ColorScale(stops)


def test_custom_render_style_reaches_prepared_frames():
    config = RenderConfig(
        max_fig_size=136, pixel_alignment=8, show_colorbar=True,
        text_color=(251, 17, 33), text_padding=9,
        background_color=(7, 8, 9), crosshair_color=(11, 199, 72), crosshair_width=3,
        colorbar_width=31, colorbar_margin=3, colorbar_min_label_width=100,
        colorbar_label_format=".1f",
    )
    source = np.zeros((16, 16))
    layout = FrameLayout(source.shape, (1, 1), (0.123, 0.987), load_render_resources(config))
    frame = render_prepared_frame(source, layout, "Title", crosshair=(8, 8))

    assert layout.output_size == (136, 136)
    assert frame.shape == (136, 280, 3)
    np.testing.assert_array_equal(frame[68, 136], config.background_color)
    assert np.any(np.all(frame == config.text_color, axis=-1))
    assert np.any(np.all(frame == config.crosshair_color, axis=-1))


@pytest.mark.parametrize("alignment", [0, -1])
def test_invalid_pixel_alignment_is_rejected(alignment):
    with pytest.raises(ValueError, match="pixel_alignment must be positive"):
        RenderConfig(pixel_alignment=alignment)


def test_colorbar_widens_output_without_changing_height():
    slice_2d = np.zeros((20, 20))
    plain = render_image_frame(slice_2d, (1.0, 1.0), RenderConfig(), (0.0, 1.0), "")
    with_colorbar = render_image_frame(
        slice_2d, (1.0, 1.0), RenderConfig(show_colorbar=True), (0.0, 1.0), ""
    )
    assert with_colorbar.shape[0] == plain.shape[0]
    assert with_colorbar.shape[1] > plain.shape[1]
    assert with_colorbar.shape[1] % 16 == 0


def test_colorbar_width_stays_aligned_with_large_labels():
    frame = render_image_frame(
        np.zeros((20, 20)), (1.0, 1.0), RenderConfig(show_colorbar=True, font_size=48),
        (-1_230_000.0, 9_870_000.0), "",
    )
    assert frame.shape[1] % 16 == 0


def test_larger_font_increases_title_footprint():
    slice_2d = np.zeros((20, 20))
    small = render_image_frame(
        slice_2d, (1.0, 1.0), RenderConfig(font_size=12), (0.0, 1.0), "Title"
    )
    large = render_image_frame(
        slice_2d, (1.0, 1.0), RenderConfig(font_size=32), (0.0, 1.0), "Title"
    )
    assert np.count_nonzero(large[:50]) > np.count_nonzero(small[:50])
