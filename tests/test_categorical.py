import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from mrplot.categorical import CategoricalColors
from mrplot.cli import cli
from mrplot.plotting import render_file
from mrplot.rendering import RenderConfig, render_image_frame
from mrplot.slicing import SliceConfig
from tests.phantoms import save_phantom


def _rgb_at(path, x, y):
    with Image.open(path) as image:
        return image.getpixel((x, y))


def test_categorical_rendering_uses_background_and_nearest_neighbour():
    config = RenderConfig(
        max_fig_size=64, background_color=(12, 23, 34),
        parcellation=CategoricalColors(background_label=-1, mapping={0: "#FF8800", 42: "#007FFF"}),
    )
    labels = np.array([[-1, 0], [42, -1]])
    frame = render_image_frame(labels, (1, 1), config, (0, 1), "")
    assert {tuple(pixel) for pixel in np.unique(frame.reshape(-1, 3), axis=0)} == {
        (12, 23, 34), (255, 136, 0), (0, 127, 255),
    }


def test_palette_accounts_for_labels_off_slice_and_in_other_frames(tmp_path, monkeypatch):
    labels = np.full((8, 8, 8, 2), 42, dtype=np.int16)
    labels[:3, :3, :3, 1] = 7
    labels[0, 0, 0, 0] = 3  # This label is absent from every displayed plane.
    path = save_phantom(tmp_path, "frames", labels)
    frames_by_plane = []

    def capture(path, frames, **kwargs):
        frames_by_plane.append(list(frames))
        return path

    monkeypatch.setattr("mrplot.plotting.write_mp4_frames", capture)
    config = RenderConfig(max_fig_size=64, parcellation=CategoricalColors())
    render_file(path, tmp_path / "out", config=config)
    assert len(frames_by_plane) == 3
    for frames in frames_by_plane:
        assert len(frames) == 2
        for frame in frames:
            np.testing.assert_array_equal(frame[32, 32], (44, 160, 44))


@pytest.mark.parametrize("four_dimensional", [False, True])
def test_overlay_colours_stay_fixed_across_frames(tmp_path, monkeypatch, four_dimensional):
    path = save_phantom(tmp_path, "base", np.zeros((8, 8, 8, 2), dtype=np.float32))
    labels = np.zeros((8, 8, 8, 2), dtype=np.int16)
    labels[2:6, 2:6, 2:6, :] = 42
    labels[0, 0, 0, 1] = 3
    outline = save_phantom(tmp_path, "outline", labels if four_dimensional else labels[..., 1])
    frames_by_plane = []

    def capture(path, frames, **kwargs):
        frames_by_plane.append(list(frames))
        return path

    monkeypatch.setattr("mrplot.plotting.write_mp4_frames", capture)
    render_file(path, tmp_path / "out", outline_path=outline, config=RenderConfig(max_fig_size=128))
    assert len(frames_by_plane) == 3
    for frames in frames_by_plane:
        assert len(frames) == 2
        for frame in frames:
            np.testing.assert_array_equal(frame[64, 32], (255, 127, 14))


@pytest.mark.parametrize(
    "table",
    [
        "# Native FreeSurfer LUT\n0 Unknown 0 0 0 0\n42 Right Cortex 205 62 78 0\n17 Hippocampus 220 216 20 0\n",
        "# Native FSLeyes LUT\n0 0 0 0 Unknown\n42 0.80392 0.24314 0.30588 Right Cortex\n",
        "# mrplot hex LUT\n42 #CD3E4E\n",
    ],
)
def test_native_and_hex_lookup_tables_via_cli(tmp_path, table):
    path = save_phantom(tmp_path, "parcels", np.full((6, 6, 6), 42, dtype=np.int16))
    lut = tmp_path / "atlas.txt"
    lut.write_text(table)
    out = tmp_path / "out"
    assert cli(["render", str(path), "--output-dir", str(out), "--parcellation-lut", str(lut)]) == 0
    assert _rgb_at(out / "parcels_axial.png", 100, 100) == (205, 62, 78)
    assert cli(["render", str(path), "--output-dir", str(out), "--outline", str(path),
                "--outline-lut", str(lut)]) == 0
    assert _rgb_at(out / "parcels_axial.png", 0, 100) == (205, 62, 78)


def test_unknown_label_in_unshown_voxel_rejects_explicit_table(tmp_path):
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[0, 0, 0] = 42
    path = save_phantom(tmp_path, "parcels", labels)
    config = RenderConfig(parcellation=CategoricalColors(mapping={7: "#E41A1C"}))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="missing labels 42"):
        render_file(path, out, config=config)
    assert not out.exists()


def test_outline_preserves_shared_border_and_matches_colour_file(tmp_path):
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[1:4, 1:7, 1:7] = 3
    labels[4:7, 1:7, 1:7] = 42
    path = save_phantom(tmp_path, "base", np.zeros_like(labels))
    outline = save_phantom(tmp_path, "parcels", labels)
    color_file = tmp_path / "outlines.txt"
    color_file.write_text("#FF0000\n#00FF00\n")

    for name, options in (
        ("inline", ["--outline-colors", "#FF0000", "#00FF00"]),
        ("file", ["--outline-colors-file", str(color_file)]),
    ):
        out = tmp_path / name
        assert cli(["render", str(path), "--output-dir", str(out), "--outline", str(outline), *options]) == 0
    for plane in ("axial", "coronal", "sagittal"):
        image = f"base_{plane}.png"
        assert (tmp_path / "inline" / image).read_bytes() == (tmp_path / "file" / image).read_bytes()

    axial = tmp_path / "inline" / "base_axial.png"
    with Image.open(axial) as image:
        x, y = image.width // 2, image.height // 2
        assert image.getpixel((x - 1, y)) == (0, 255, 0)
        assert image.getpixel((x, y)) == (255, 0, 0)
        assert image.getpixel((x + 1, y)) == (0, 0, 0)


def test_outline_respects_grid_and_mask_slice_selection(tmp_path):
    labels = np.zeros((8, 10, 12), dtype=np.uint8)
    labels[0:3, 0:3, 0:3] = 1
    path = save_phantom(tmp_path, "base", np.zeros_like(labels))
    mask = save_phantom(tmp_path, "mask", labels)
    transform = np.array([[1, -1], [2, 1], [0, 1]])
    reoriented = nib.Nifti1Image(labels, np.eye(4)).as_reoriented(transform)
    outline = tmp_path / "reoriented.nii.gz"
    nib.save(reoriented, outline)

    config = RenderConfig(max_fig_size=96, slicing=SliceConfig(crop=True, padding=1))
    args = dict(mask_path=mask, config=config, outline_colors=CategoricalColors(colors=["#FF8800"]))
    reference = render_file(path, tmp_path / "reference", outline_path=mask, **args)
    actual = render_file(path, tmp_path / "aligned", outline_path=outline, **args)
    for expected_path, actual_path in zip(reference, actual):
        with Image.open(expected_path) as expected, Image.open(actual_path) as image:
            np.testing.assert_array_equal(image, expected)
            assert np.any(np.all(np.asarray(image) == (255, 136, 0), axis=-1))


def test_crop_uses_configured_parcellation_background(tmp_path):
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    mask = np.zeros_like(labels)
    mask[2:6, 2:6, 2:6] = 1
    path = save_phantom(tmp_path, "parcels", labels)
    mask_path = save_phantom(tmp_path, "mask", mask)
    config = RenderConfig(
        max_fig_size=96, slicing=SliceConfig(crop=True, padding=1),
        parcellation=CategoricalColors(background_label=-1, mapping={0: "#FF8800"}),
    )
    _, _, axial = render_file(path, tmp_path / "out", config=config, mask_path=mask_path)
    assert _rgb_at(axial, 48, 48) == (255, 136, 0)
    assert _rgb_at(axial, 0, 48) == (0, 0, 0)


def test_outline_value_limit_checks_every_voxel(tmp_path):
    labels = np.zeros((10, 10, 6), dtype=np.int16)
    labels[..., 1:] = np.arange(500, dtype=np.int16).reshape(10, 10, 5)
    path = save_phantom(tmp_path, "base", np.zeros_like(labels))
    accepted = save_phantom(tmp_path, "accepted", labels)
    config = RenderConfig(max_fig_size=64)
    assert len(render_file(path, tmp_path / "out", config=config, outline_path=accepted)) == 3

    labels[0, 0, 0] = 500  # Outside each displayed slice.
    rejected = save_phantom(tmp_path, "rejected", labels)
    with pytest.raises(ValueError, match="501 unique voxel values.*limit is 500"):
        render_file(path, tmp_path / "absent", config=config, outline_path=rejected)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize(
    ("colors", "error"),
    [
        (["orange"], "Invalid hex colour"),
        (["#FF8800", "#0088FF"], "expects 1 foreground region colours, got 2"),
        ([], "expects 1 foreground region colours, got 0"),
    ],
)
def test_invalid_outline_colours_fail_before_writing(tmp_path, colors, error):
    path = save_phantom(tmp_path, "base", np.zeros((8, 8, 8), dtype=np.float32))
    outline = save_phantom(tmp_path, "outline", np.ones((8, 8, 8), dtype=np.int16))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=error):
        render_file(path, out, outline_path=outline, outline_colors=CategoricalColors(colors=colors))
    assert not out.exists()


@pytest.mark.parametrize(
    ("shape", "offset", "value", "error"),
    [
        ((8, 8, 8), 0, 0.5, "finite integer voxel labels"),
        ((8, 8, 8), 0, np.nan, "finite integer voxel labels"),
        ((8, 8, 8), 1, 1, "geometry must match"),
        ((4, 8, 8), 0, 1, "geometry must match"),
        ((8, 8, 8, 1), 0, 1, "same number of frames"),
        ((8, 8), 0, 1, "must be 3D or 4D"),
    ],
)
def test_invalid_outline_volumes_fail_before_writing(tmp_path, shape, offset, value, error):
    path = save_phantom(tmp_path, "base", np.zeros((8, 8, 8), dtype=np.float32))
    labels = np.ones(shape, dtype=np.float32)
    labels.flat[0] = value
    affine = np.eye(4)
    affine[0, 3] = offset
    outline = save_phantom(tmp_path, "outline", labels, affine)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=error):
        render_file(path, out, outline_path=outline)
    assert not out.exists()
