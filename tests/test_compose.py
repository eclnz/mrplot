import numpy as np
import pytest
from PIL import Image

from mrplot.cli import cli
from mrplot.compose import Panel, compose_files, panels_from_directories
from mrplot.slicing import Plane


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
    panels = panels_from_directories([("T1", tmp_path / "t1"), ("FAST", tmp_path / "fast")], [Plane.AXIAL, Plane.CORONAL])
    assert [(panel.label, panel.path.parent.name, panel.path.name) for panel in panels] == [
        ("T1", "t1", "scan_axial.png"), ("FAST", "fast", "scan_axial.png"),
        ("T1", "t1", "scan_coronal.png"), ("FAST", "fast", "scan_coronal.png"),
    ]
    with pytest.raises(ValueError, match="sagittal"):
        panels_from_directories([("T1", tmp_path / "t1")], [Plane.SAGITTAL])


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


def test_grid_layout_geometry():
    from mrplot.compose import _GridLayout

    grid = _GridLayout(count=5, columns=2, tile_size=(40, 30), label_height=10, spacing=(5, 3), title_height=20)
    assert grid.cell_size == (50, 46)
    assert grid.size == (100, 20 + 3 * 46)
    assert grid.tile_origin(0) == (5, 23)
    assert grid.tile_origin(3) == (55, 20 + 46 + 3)
    assert _GridLayout(count=1, columns=4, tile_size=(8, 8), label_height=0, spacing=(0, 0)).size == (8, 8)


def _frames_folder(folder, plane_counts, stem="raw_delays", color=(200, 0, 0)):
    for plane, count in plane_counts.items():
        for index in range(count):
            _png(folder / f"{stem}_{plane}_vol-{index:04d}.png", (6, 6), (color[0], index * 40, color[2]))


def test_frame_folder_matches_the_hand_listed_grid(tmp_path):
    """Acceptance for issue #9: three planes x (three delays + two single maps)."""
    import numpy as np
    from tests.phantoms import save_phantom

    delays = save_phantom(tmp_path, "raw_delays", np.random.default_rng(0).random((10, 12, 8, 3)).astype(np.float32))
    console = save_phantom(tmp_path, "scanner_cbf", np.random.default_rng(1).random((10, 12, 8)).astype(np.float32))
    basil = save_phantom(tmp_path, "perfusion_calib", np.random.default_rng(2).random((10, 12, 8)).astype(np.float32))
    common = ["--fps", "1"]
    assert cli(["render", str(delays), "--output-dir", str(tmp_path / "delays"), "--frames-as-png", *common]) == 0
    assert cli(["render", str(console), "--output-dir", str(tmp_path / "console"), *common]) == 0
    assert cli(["render", str(basil), "--output-dir", str(tmp_path / "basil"), *common]) == 0

    automatic = tmp_path / "rows.png"
    assert cli([
        "compose", "--output", str(automatic), "--rows", "axial", "coronal", "sagittal",
        "--panel", "TI 1.57 s,TI 2.46 s,TI 4.5 s", str(tmp_path / "delays"),
        "--panel", "Console", str(tmp_path / "console"), "--panel", "BASIL", str(tmp_path / "basil"),
    ]) == 0

    manual_args = ["compose", "--output", str(tmp_path / "manual.png"), "--columns", "5"]
    for plane in ("axial", "coronal", "sagittal"):
        for index, label in enumerate(("TI 1.57 s", "TI 2.46 s", "TI 4.5 s")):
            manual_args += ["--panel", label, str(tmp_path / "delays" / f"raw_delays_{plane}_vol-{index:04d}.png")]
        manual_args += ["--panel", "Console", str(tmp_path / "console" / f"scanner_cbf_{plane}.png")]
        manual_args += ["--panel", "BASIL", str(tmp_path / "basil" / f"perfusion_calib_{plane}.png")]
    assert cli(manual_args) == 0

    with Image.open(automatic) as a, Image.open(tmp_path / "manual.png") as b:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_frame_labels_selection_and_sessions(tmp_path):
    for session, count in (("ses-01", 3), ("ses-02", 4)):
        _frames_folder(tmp_path / session, {"axial": count, "coronal": count})
    folders = [("TI", tmp_path / "ses-01"), (None, tmp_path / "ses-02")]

    panels = panels_from_directories(folders, [Plane.AXIAL, Plane.CORONAL])
    # Sessions with different frame counts sit side by side: 3 + 4 columns per row.
    assert len(panels) == 14
    assert [panel.label for panel in panels[:7]] == ["TI 0", "TI 1", "TI 2", None, None, None, None]
    assert panels[7].path.name == "raw_delays_coronal_vol-0000.png"

    chosen = panels_from_directories(folders, [Plane.AXIAL], frames=[0, 2])
    assert [(panel.label, panel.path.name) for panel in chosen] == [
        ("TI 0", "raw_delays_axial_vol-0000.png"), ("TI 2", "raw_delays_axial_vol-0002.png"),
        (None, "raw_delays_axial_vol-0000.png"), (None, "raw_delays_axial_vol-0002.png"),
    ]
    listed = panels_from_directories([(r"Early\, short,Mid,Late", tmp_path / "ses-01")], [Plane.AXIAL])
    assert [panel.label for panel in listed] == ["Early, short", "Mid", "Late"]


@pytest.mark.parametrize(
    ("setup", "kwargs", "message"),
    [
        (lambda d: _frames_folder(d, {"axial": 3}), {"label": "A,B"}, "2 labels for 3 frames"),
        (lambda d: _frames_folder(d, {"axial": 3, "coronal": 2}), {"rows": [Plane.AXIAL, Plane.CORONAL]}, "3 images for axial but 2"),
        (lambda d: _frames_folder(d, {"axial": 2}), {"frames": [0, 5]}, "no axial frame 5"),
    ],
)
def test_frame_folder_errors(tmp_path, setup, kwargs, message):
    folder = tmp_path / "frames"
    setup(folder)
    rows = kwargs.pop("rows", [Plane.AXIAL])
    label = kwargs.pop("label", None)
    with pytest.raises(ValueError, match=message):
        panels_from_directories([(label, folder)], rows, **kwargs)


def test_cli_frames_and_rows_parsing(tmp_path, capsys):
    from mrplot.cli import _build_parser

    parsed = _build_parser().parse_args([
        "compose", "--output", "x.png", "--rows", "axial,coronal", "sagittal", "--frames", "0,2-4, 7",
        "--images", "a",
    ])
    assert parsed.rows == [[Plane.AXIAL, Plane.CORONAL], [Plane.SAGITTAL]]
    assert parsed.frames == [0, 2, 3, 4, 7]
    for bad in (["--frames", "3-1"], ["--frames", "a"], ["--frames", ","], ["--rows", "axail"]):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["compose", "--output", "x.png", "--images", "a", *bad])
    assert "choose from sagittal, coronal, axial" in capsys.readouterr().err


def test_cli_rows_rejects_mismatched_columns_and_frames_without_rows(tmp_path, capsys):
    _frames_folder(tmp_path / "frames", {"axial": 3})
    out = str(tmp_path / "x.png")
    assert cli(["compose", "--output", out, "--rows", "axial", "--columns", "2", "--images", str(tmp_path / "frames")]) == 2
    assert "--rows gives 3 columns" in capsys.readouterr().err
    assert cli(["compose", "--output", out, "--rows", "axial", "--columns", "3", "--images", str(tmp_path / "frames")]) == 0
    assert cli(["compose", "--output", out, "--frames", "0", "--images", str(tmp_path / "frames" / "raw_delays_axial_vol-0000.png")]) == 2
    assert "--frames requires --rows" in capsys.readouterr().err
