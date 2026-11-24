import numpy as np
import pytest
from PIL import Image

from mrplot.cli import cli
from mrplot.compose import Panel, compose_files, panels_from_directories


def _png(path, size, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _rgb(path, x, y):
    with Image.open(path) as image:
        return image.convert("RGB").getpixel((x, y))


def test_unlabelled_grid_centres_smaller_images(tmp_path):
    red = _png(tmp_path / "red.png", (40, 30), (255, 0, 0))
    blue = _png(tmp_path / "blue.png", (20, 10), (0, 0, 255))
    green = _png(tmp_path / "green.png", (40, 30), (0, 255, 0))

    out = compose_files([red, blue, green], tmp_path / "grid.png", columns=2, spacing=(5, 3))

    with Image.open(out) as image:
        # Two columns of 40 + 2*5 wide tiles; two rows of 30 + 2*3 high tiles.
        assert image.size == (100, 72)
    assert _rgb(out, 5, 3) == (255, 0, 0)
    assert _rgb(out, 75, 18) == (0, 0, 255)  # Centre of the second tile.
    assert _rgb(out, 56, 4) == (0, 0, 0)  # The smaller image leaves background around it.
    assert _rgb(out, 5, 39) == (0, 255, 0)
    assert _rgb(out, 75, 54) == (0, 0, 0)  # The unused cell stays empty.


def test_labels_and_title_add_text_bands(tmp_path):
    image = _png(tmp_path / "a.png", (300, 300), (80, 80, 80))
    plain = compose_files([image, image], tmp_path / "plain.png", spacing=(0, 0))
    labelled = compose_files(
        [Panel(image, "T1"), Panel(image, "FAST")], tmp_path / "labelled.png",
        title="sub-01", spacing=(0, 0), text_color=(255, 255, 0),
    )
    with Image.open(plain) as a, Image.open(labelled) as b:
        assert a.size == (600, 300)
        assert b.width == 600 and b.height > 300
        pixels = np.asarray(b)
    text = (pixels[..., 0] > 120) & (pixels[..., 1] > 120) & (pixels[..., 2] < 60)
    yellow_rows = np.flatnonzero(text.any(axis=1))
    image_rows = np.flatnonzero(np.all(pixels == (80, 80, 80), axis=-1).any(axis=1))
    assert yellow_rows.min() < image_rows.min()  # Title above.
    assert yellow_rows.max() > image_rows.max()  # Labels below.


def test_default_font_scales_with_tile_size(tmp_path):
    heights = []
    for size in (300, 1200):
        image = _png(tmp_path / f"{size}.png", (size, size), (80, 80, 80))
        out = compose_files([Panel(image, "Label")], tmp_path / f"out{size}.png", spacing=(0, 0))
        with Image.open(out) as composed:
            heights.append(composed.height - size)
    assert heights[1] > 3 * heights[0]


def test_transparent_pixels_take_the_background(tmp_path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 0)).save(path)
    out = compose_files([path], tmp_path / "out.png", spacing=(0, 0), background_color=(1, 2, 3))
    assert _rgb(out, 5, 5) == (1, 2, 3)


def test_invalid_arguments(tmp_path):
    image = _png(tmp_path / "a.png", (4, 4), (0, 0, 0))
    with pytest.raises(ValueError, match="At least one"):
        compose_files([], tmp_path / "out.png")
    with pytest.raises(ValueError, match="columns"):
        compose_files([image], tmp_path / "out.png", columns=0)
    with pytest.raises(ValueError, match="spacing"):
        compose_files([image], tmp_path / "out.png", spacing=(-1, 0))


def test_panels_from_directories(tmp_path):
    for folder in ("t1", "fast"):
        for plane in ("axial", "coronal"):
            _png(tmp_path / folder / f"scan_{plane}.png", (4, 4), (0, 0, 0))
    panels = panels_from_directories([("T1", tmp_path / "t1"), ("FAST", tmp_path / "fast")], ["axial", "coronal"])
    assert [(panel.label, panel.path.parent.name, panel.path.name) for panel in panels] == [
        ("T1", "t1", "scan_axial.png"), ("FAST", "fast", "scan_axial.png"),
        ("T1", "t1", "scan_coronal.png"), ("FAST", "fast", "scan_coronal.png"),
    ]
    with pytest.raises(ValueError, match="sagittal"):
        panels_from_directories([("T1", tmp_path / "t1")], ["sagittal"])


def test_cli_compose_keeps_panel_order(tmp_path, capsys):
    red = _png(tmp_path / "red.png", (20, 20), (255, 0, 0))
    blue = _png(tmp_path / "blue.png", (20, 20), (0, 0, 255))
    out = tmp_path / "out" / "figure.png"
    status = cli([
        "compose", "--output", str(out), "--columns", "1", "--spacing", "0", "0",
        "--images", str(blue), "--panel", "Red", str(red), "--background-color", "#102030",
    ])
    assert status == 0
    assert capsys.readouterr().out.strip() == str(out)
    assert _rgb(out, 10, 10) == (0, 0, 255)
    with Image.open(out) as image:
        assert image.height > 40
        last_row = image.height - 1
    assert _rgb(out, 0, last_row) == (16, 32, 48)


def test_cli_compose_rows_from_render_folders(tmp_path):
    for folder, color in (("t1", (255, 0, 0)), ("seg", (0, 255, 0))):
        for plane in ("axial", "coronal", "sagittal"):
            _png(tmp_path / folder / f"scan_{plane}.png", (10, 10), color)
    out = tmp_path / "grid.png"
    assert cli([
        "compose", "--output", str(out), "--rows", "axial,coronal", "sagittal", "--spacing", "0", "0",
        "--images", str(tmp_path / "t1"), str(tmp_path / "seg"),
    ]) == 0
    with Image.open(out) as image:
        assert image.size == (20, 30)
    assert _rgb(out, 5, 25) == (255, 0, 0)
    assert _rgb(out, 15, 25) == (0, 255, 0)


def test_cli_compose_errors(tmp_path, capsys):
    assert cli(["compose", "--output", str(tmp_path / "x.png")]) == 2
    assert "at least one" in capsys.readouterr().err
    assert cli(["compose", "--output", str(tmp_path / "x.png"), "--images", str(tmp_path / "missing.png")]) == 1
