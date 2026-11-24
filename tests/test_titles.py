import numpy as np
import pytest
from PIL import Image, ImageDraw

from mrplot.cli import cli
from mrplot.colors import GRAY
from mrplot.overlays import ImageOverlay, LabelOverlay
from mrplot.plotting import layer_label, plane_title, render_file
from mrplot.rendering import RenderConfig, load_font, render_image_frame, wrap_title
from mrplot.slicing import Plane
from tests.phantoms import save_phantom


def _titles(monkeypatch):
    titles = []

    def capture(slice_2d, layout, title, **kwargs):
        titles.append(title)
        return np.zeros((*layout.output_size[::-1], 3), dtype=np.uint8)

    monkeypatch.setattr("mrplot.plotting.render_prepared_frame", capture)
    return titles


def test_titles_name_layers_in_drawing_order(tmp_path, monkeypatch):
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    base = save_phantom(tmp_path, "T1_biascorr", labels.astype(np.float32))
    seg = save_phantom(tmp_path, "T1_fast_seg", labels)
    cbf = save_phantom(tmp_path, "cbf", labels.astype(np.float32))
    titles = _titles(monkeypatch)

    render_file(
        base, tmp_path / "out", config=RenderConfig(max_fig_size=64),
        overlays=[ImageOverlay(cbf), LabelOverlay(seg), LabelOverlay(seg, fill_opacity=0.5, line_opacity=0)],
    )

    layers = "T1_biascorr + cbf (overlay) + T1_fast_seg (outline) + T1_fast_seg (mask)"
    assert titles == [f"{layers} {plane} Slice" for plane in ("Sagittal", "Coronal", "Axial")]


def test_plain_render_keeps_the_base_title():
    assert plane_title(RenderConfig(), Plane.AXIAL, base_name="scan") == "scan Axial Slice"


def test_title_template_and_no_title():
    config = RenderConfig(title="{name} [{plane}] with {layers}")
    assert plane_title(config, Plane.CORONAL, base_name="t1", layer_labels=["seg (outline)"]) == (
        "t1 [Coronal] with t1 + seg (outline)"
    )
    assert plane_title(RenderConfig(title="Fixed"), Plane.AXIAL, base_name="t1") == "Fixed"
    assert plane_title(RenderConfig(show_title=False), Plane.AXIAL, base_name="t1") == ""


def test_invalid_title_template_is_rejected():
    with pytest.raises(ValueError, match="title template"):
        RenderConfig(title="{subject}")


def test_layer_label_strips_nifti_suffix():
    assert layer_label(LabelOverlay("/data/synthseg.nii.gz")) == "synthseg (outline)"


def test_wrap_title_fits_width_and_respects_line_limit():
    font = load_font(None, 20)
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title = "T1_biascorr + T1_fast_seg (outline) + synthseg (outline) Axial Slice"
    lines = wrap_title(draw, title, 200, font)
    assert len(lines) > 1
    assert all(draw.textlength(line, font=font) <= 200 for line in lines)
    assert " ".join(lines) == title

    limited = wrap_title(draw, title, 200, font, max_lines=2)
    assert len(limited) == 2 and limited[-1].endswith("...")
    assert all(draw.textlength(line, font=font) <= 200 for line in limited)


def test_long_title_wraps_instead_of_overflowing():
    config = RenderConfig(max_fig_size=256, color_scale=GRAY)
    data = np.zeros((32, 32))
    short = render_image_frame(data, (1.0, 1.0), config, (0.0, 1.0), "Axial")
    long = render_image_frame(data, (1.0, 1.0), config, (0.0, 1.0), "word " * 20)
    rows_with_text = lambda frame: np.flatnonzero(frame.any(axis=(1, 2)))
    assert rows_with_text(long).max() > rows_with_text(short).max()
    assert rows_with_text(long).max() < 256 // 4


def test_cli_title_options(tmp_path, monkeypatch):
    base = save_phantom(tmp_path, "scan", np.ones((6, 6, 6), dtype=np.float32))
    titles = _titles(monkeypatch)
    assert cli(["render", str(base), "--output-dir", str(tmp_path / "a"), "--title", "{plane} QC"]) == 0
    assert cli(["render", str(base), "--output-dir", str(tmp_path / "b"), "--no-title"]) == 0
    assert titles == ["Sagittal QC", "Coronal QC", "Axial QC", "", "", ""]
    assert cli(["render", str(base), "--output-dir", str(tmp_path / "c"), "--title", "{oops}"]) == 2
