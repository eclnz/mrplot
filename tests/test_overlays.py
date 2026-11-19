import numpy as np
import pytest
from PIL import Image

from mrplot.categorical import CategoricalColors
from mrplot.cli import cli
from mrplot.colors import BLUE_RED
from mrplot.overlays import ImageOverlay, LabelOverlay
from mrplot.plotting import render_file
from mrplot.rendering import Interpolation, RenderConfig
from tests.phantoms import save_phantom


def test_roi_line_and_fill_opacity_blend_independently(tmp_path):
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    base = save_phantom(tmp_path, "base", np.zeros_like(labels))
    mask = save_phantom(tmp_path, "mask", labels)
    _, _, axial = render_file(base, tmp_path / "out", config=RenderConfig(max_fig_size=128), overlays=[
        LabelOverlay(mask, colors=CategoricalColors(colors=["#FF0000"]), fill_opacity=0.25, line_opacity=0.5),
    ])
    with Image.open(axial) as view:
        assert view.getpixel((31, 64)) == (0, 0, 0)
        assert view.getpixel((32, 64)) == (159, 0, 0)
        assert view.getpixel((64, 64)) == (64, 0, 0)


def test_cli_draws_repeated_images_and_rois_in_argument_order(tmp_path):
    base = save_phantom(tmp_path, "base", np.zeros((8, 8, 8), dtype=np.uint8))
    image = save_phantom(tmp_path, "constant", np.ones((8, 8, 8), dtype=np.float32))
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    mask = save_phantom(tmp_path, "mask", labels)
    lut = tmp_path / "labels.txt"
    lut.write_text("1 #0000FF\n")
    out = tmp_path / "out"
    assert cli([
        "render", str(base), "--output-dir", str(out),
        "--overlay", str(image), "--overlay-opacity", "0.5", "--overlay-color-scale", "blue-red",
        "--outline", str(mask), "--outline-colors", "#00FF00", "--mask-opacity", "0.5", "--outline-opacity", "0",
        "--overlay", str(image), "--overlay-opacity", "0.5", "--overlay-color-scale", "gray",
        "--outline", str(mask), "--outline-lut", str(lut),
    ]) == 0
    with Image.open(out / "base_axial.png") as view:
        centre = view.width // 2
        assert view.getpixel((centre, centre)) == (160, 192, 128)
        assert view.getpixel((view.width // 4, centre)) == (0, 0, 255)
        assert view.getpixel((view.width // 8, centre)) == (192, 128, 128)


def test_continuous_overlay_treats_zero_and_nonfinite_voxels_as_transparent(tmp_path):
    base = save_phantom(tmp_path, "base", np.zeros((8, 8, 8), dtype=np.uint8))
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[2:6, 2:6, 2:6] = 1.0
    data[4:6, 2:6, 2:6] = np.nan
    overlay = save_phantom(tmp_path, "overlay", data)
    config = RenderConfig(max_fig_size=128, interpolation=Interpolation.NEAREST)
    _, _, axial = render_file(base, tmp_path / "out", config=config, overlays=[
        ImageOverlay(overlay, color_scale=BLUE_RED, opacity=1),
    ])
    with Image.open(axial) as view:
        assert view.getpixel((80, 64)) == (255, 0, 0)
        assert view.getpixel((48, 64)) == (0, 0, 0)
        assert view.getpixel((16, 64)) == (0, 0, 0)


def test_ordered_layers_reuse_static_roi_and_advance_4d_image(tmp_path, monkeypatch):
    shape = 8, 8, 8
    base = save_phantom(tmp_path, "base", np.zeros((*shape, 2), dtype=np.float32))
    data = np.ones((*shape, 2), dtype=np.float32)
    data[..., 1] = 2
    overlay = save_phantom(tmp_path, "dynamic", data)
    region = np.zeros(shape, dtype=np.uint8)
    region[2:6, 2:6, 2:6] = 1
    mask = save_phantom(tmp_path, "mask", region)
    frames = []

    def capture(path, source, **kwargs):
        frames.append(list(source))
        return path

    monkeypatch.setattr("mrplot.plotting.write_mp4_frames", capture)
    render_file(base, tmp_path / "out", config=RenderConfig(max_fig_size=128), overlays=[
        ImageOverlay(overlay, opacity=1, color_scale=BLUE_RED, intensity_min=1, intensity_max=2),
        LabelOverlay(mask, colors=CategoricalColors(colors=["#00FF00"])),
    ])
    assert len(frames) == 3
    for first, second in frames:
        np.testing.assert_array_equal(first[64, 16], (0, 0, 255))
        np.testing.assert_array_equal(second[64, 16], (255, 0, 0))
        np.testing.assert_array_equal(first[64, 32], (0, 255, 0))
        np.testing.assert_array_equal(second[64, 32], (0, 255, 0))


@pytest.mark.parametrize("flag, value", [
    ("--overlay-opacity", "nan"), ("--outline-opacity", "-0.2"), ("--mask-opacity", "1.2"),
])
def test_cli_rejects_invalid_opacity_before_writing(tmp_path, capsys, flag, value):
    path = save_phantom(tmp_path, "base", np.zeros((4, 4, 4), dtype=np.uint8))
    out = tmp_path / "out"
    layer = "--overlay" if flag == "--overlay-opacity" else "--outline"
    assert cli(["render", str(path), "--output-dir", str(out), layer, str(path), flag, value]) == 2
    assert "opacity must be between 0 and 1" in capsys.readouterr().err
    assert not out.exists()


def test_bad_geometry_in_later_layer_fails_before_writing(tmp_path):
    data = np.ones((4, 4, 4), dtype=np.uint8)
    base = save_phantom(tmp_path, "base", data)
    affine = np.eye(4)
    affine[0, 3] = 1
    shifted = save_phantom(tmp_path, "shifted", data, affine)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="shifted.*geometry must match"):
        render_file(base, out, overlays=[ImageOverlay(base), ImageOverlay(shifted)])
    assert not out.exists()
