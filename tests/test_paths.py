import pytest
from PIL import Image

from mrplot.paths import find_plane_outputs, frame_output_path, plane_output_stem
from mrplot.slicing import Plane


def _touch_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


def test_output_names():
    stem = plane_output_stem("out", "scan", Plane.AXIAL)
    assert stem.replace("\\", "/") == "out/scan_axial"
    assert frame_output_path(stem, 3, 10).endswith("scan_axial_vol-0003.png")
    assert frame_output_path(stem, 3, 12000).endswith("scan_axial_vol-00003.png")


@pytest.mark.parametrize("frame_count", [1, 3, 12])
def test_find_plane_outputs_reads_back_what_render_names(tmp_path, frame_count):
    for plane in Plane:
        stem = plane_output_stem(tmp_path / "frames", "raw.delays", plane)
        for index in range(frame_count):
            _touch_png(tmp_path / frame_output_path(stem, index, frame_count))
        _touch_png(tmp_path / f"{plane_output_stem(tmp_path / 'single', 'scan_axial', plane)}.png")

    for plane in Plane:
        frames = find_plane_outputs(tmp_path / "frames", plane)
        assert frames.single is None
        assert list(frames.frames) == list(range(frame_count))
        assert all(path.name.startswith(f"raw.delays_{plane.value}_vol-") for path in frames.frames.values())

        single = find_plane_outputs(tmp_path / "single", plane)
        assert single.single.name == f"scan_axial_{plane.value}.png"
        assert not single.frames


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (["a_axial.png", "b_axial.png"], "Expected one"),
        (["scan_axial.png", "scan_axial_vol-0000.png"], "has both"),
        (["a_axial_vol-0000.png", "b_axial_vol-0000.png"], "several renders"),
        (["scan_coronal.png"], "found none"),
    ],
)
def test_find_plane_outputs_rejects_ambiguous_folders(tmp_path, files, message):
    for name in files:
        _touch_png(tmp_path / name)
    with pytest.raises(ValueError, match=message):
        find_plane_outputs(tmp_path, Plane.AXIAL)


def test_find_plane_outputs_requires_a_directory(tmp_path):
    with pytest.raises(ValueError, match="Not a directory"):
        find_plane_outputs(tmp_path / "missing", Plane.AXIAL)
