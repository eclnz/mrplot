import os
import shutil

import pytest

from mrplot.cli import _build_parser, cli
from mrplot.rendering import Interpolation, RenderConfig


def _base_name(path) -> str:
    return os.path.basename(str(path)).split(".nii")[0]


def test_render_3d_produces_pngs(sample_3d_nifti, tmp_path, capsys):
    output_dir = tmp_path / "out"
    exit_code = cli(["render", str(sample_3d_nifti), "--output-dir", str(output_dir)])
    assert exit_code == 0

    base = _base_name(sample_3d_nifti)
    stdout = capsys.readouterr().out
    for plane in ("axial", "coronal", "sagittal"):
        path = output_dir / f"{base}_{plane}.png"
        assert path.exists()
        assert path.stat().st_size > 0
        assert str(path) in stdout


def test_render_directory_mirrors_paths_and_skips_nested_output(sample_3d_nifti, tmp_path, capsys):
    input_dir = tmp_path / "scans"
    output_dir = input_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "not-a-scan.nii").touch()
    (input_dir / "notes.txt").touch()
    for directory in ("patient 1", "patient 2"):
        source_dir = input_dir / directory
        source_dir.mkdir()
        shutil.copyfile(sample_3d_nifti, source_dir / "visit.nii.followup.nii.gz")

    status = cli([
        "render-directory", str(input_dir), "--output-dir", str(output_dir),
        "--colorbar", "--font-size", "24",
    ])

    assert status == 0
    expected = {
        output_dir / directory / f"visit.nii.followup_{plane}.png"
        for directory in ("patient 1", "patient 2")
        for plane in ("axial", "coronal", "sagittal")
    }
    assert set(output_dir.rglob("*.png")) == expected
    assert set(capsys.readouterr().out.splitlines()) == {str(path) for path in expected}


def test_render_directory_rejects_colliding_output_names_before_overwriting(
    sample_3d_nifti, tmp_path, capsys,
):
    import nibabel as nib

    input_dir = tmp_path / "scans"
    input_dir.mkdir()
    nib.load(sample_3d_nifti).to_filename(input_dir.joinpath("scan.nii"))
    input_dir.joinpath("scan.nii.gz").touch()
    output_dir = tmp_path / "output"

    assert cli(["render-directory", str(input_dir), "--output-dir", str(output_dir)]) == 2
    assert "output name collision" in capsys.readouterr().err
    assert {path.name for path in output_dir.iterdir()} == {
        f"scan_{plane}.png" for plane in ("sagittal", "coronal", "axial")
    }


@pytest.mark.parametrize("command", ["render", "render-directory"])
def test_cli_reports_invalid_nifti(command, tmp_path, capsys):
    input_dir = tmp_path / "scans"
    input_dir.mkdir()
    invalid_scan = input_dir / "invalid.nii"
    invalid_scan.write_text("invalid scan")
    source = invalid_scan if command == "render" else input_dir
    output_dir = tmp_path / "output"

    assert cli([command, str(source), "--output-dir", str(output_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert str(invalid_scan) in captured.err
    assert not output_dir.exists()


def test_render_4d_produces_mp4s(sample_4d_nifti, tmp_path):
    output_dir = tmp_path / "out"
    exit_code = cli(
        ["render", str(sample_4d_nifti), "--output-dir", str(output_dir), "--fps", "5"]
    )
    assert exit_code == 0

    base = _base_name(sample_4d_nifti)
    for plane in ("axial", "coronal", "sagittal"):
        path = output_dir / f"{base}_{plane}.mp4"
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_5d_is_rejected_clearly(sample_5d_nifti, tmp_path, capsys):
    exit_code = cli(["render", str(sample_5d_nifti), "--output-dir", str(tmp_path / "out")])
    assert exit_code != 0
    assert "5D" in capsys.readouterr().err


def test_render_with_mask_and_crop(sample_3d_nifti, sample_mask, tmp_path):
    output_dir = tmp_path / "out"
    exit_code = cli(
        [
            "render",
            str(sample_3d_nifti),
            "--mask",
            str(sample_mask),
            "--crop",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    base = _base_name(sample_3d_nifti)
    for plane in ("axial", "coronal", "sagittal"):
        assert (output_dir / f"{base}_{plane}.png").exists()


def test_render_crop_without_mask_errors(sample_3d_nifti, tmp_path, capsys):
    exit_code = cli(
        ["render", str(sample_3d_nifti), "--crop", "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code != 0
    assert "--mask" in capsys.readouterr().err


def test_render_accepts_interpolation_flag(sample_3d_nifti, tmp_path):
    output_dir = tmp_path / "out"
    exit_code = cli(
        [
            "render",
            str(sample_3d_nifti),
            "--interpolation",
            "nearest",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    base = _base_name(sample_3d_nifti)
    assert (output_dir / f"{base}_axial.png").exists()


def test_interpolation_defaults_to_bicubic():
    args = _build_parser().parse_args(["render", "scan.nii", "--output-dir", "out"])
    assert args.interpolation == Interpolation.BICUBIC.value
    assert RenderConfig().interpolation is Interpolation.BICUBIC


def test_render_rejects_unknown_interpolation(sample_3d_nifti, tmp_path):
    # argparse's own `choices` validation exits directly (SystemExit), it
    # never reaches rendering.
    with pytest.raises(SystemExit):
        cli(
            [
                "render",
                str(sample_3d_nifti),
                "--interpolation",
                "bogus",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_render_accepts_crosshairs_and_colorbar_flags(sample_3d_nifti, tmp_path):
    output_dir = tmp_path / "out"
    exit_code = cli(
        [
            "render",
            str(sample_3d_nifti),
            "--crosshairs",
            "--colorbar",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    base = _base_name(sample_3d_nifti)
    assert (output_dir / f"{base}_axial.png").exists()


def test_font_size_defaults_to_20_and_can_be_changed(sample_3d_nifti, tmp_path):
    import numpy as np
    from PIL import Image

    assert _build_parser().parse_args(["render", "scan.nii", "--output-dir", "out"]).font_size == 20

    default_dir, larger_dir = tmp_path / "default", tmp_path / "larger"
    cli(["render", str(sample_3d_nifti), "--output-dir", str(default_dir)])
    cli(
        [
            "render", str(sample_3d_nifti), "--font-size", "32",
            "--output-dir", str(larger_dir),
        ]
    )

    base = _base_name(sample_3d_nifti)
    default = np.array(Image.open(default_dir / f"{base}_axial.png"))
    larger = np.array(Image.open(larger_dir / f"{base}_axial.png"))
    assert default.shape == larger.shape
    assert not np.array_equal(default, larger)


def test_font_size_must_be_positive(sample_3d_nifti, tmp_path, capsys):
    exit_code = cli(
        ["render", str(sample_3d_nifti), "--font-size", "0", "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code == 2
    assert "font_size must be positive" in capsys.readouterr().err


def test_colorbar_flag_widens_output(sample_3d_nifti, tmp_path):
    from PIL import Image

    plain_dir, colorbar_dir = tmp_path / "plain", tmp_path / "colorbar"
    cli(["render", str(sample_3d_nifti), "--output-dir", str(plain_dir)])
    cli(["render", str(sample_3d_nifti), "--colorbar", "--output-dir", str(colorbar_dir)])

    base = _base_name(sample_3d_nifti)
    plain = Image.open(plain_dir / f"{base}_axial.png")
    with_colorbar = Image.open(colorbar_dir / f"{base}_axial.png")
    assert with_colorbar.size[0] > plain.size[0]
    assert with_colorbar.size[1] == plain.size[1]


def test_cli_applies_selected_color_scale(sample_3d_nifti, tmp_path):
    import numpy as np
    from PIL import Image

    assert cli([
        "render", str(sample_3d_nifti), "--output-dir", str(tmp_path),
        "--color-scale", "heat", "--colorbar",
    ]) == 0
    path = tmp_path.joinpath(f"{_base_name(sample_3d_nifti)}_axial.png")
    with Image.open(path) as image:
        frame = np.array(image)
    assert np.any(frame[..., 0] > frame[..., 1])


def test_min_max_flags_narrow_the_intensity_window(sample_3d_nifti, tmp_path):
    import numpy as np
    from PIL import Image

    # sample_3d_nifti's data lies in [0, 1); widening the window to [-5, 5]
    # should compress the rendered output into a narrower grayscale range.
    auto_dir, manual_dir = tmp_path / "auto", tmp_path / "manual"
    cli(["render", str(sample_3d_nifti), "--output-dir", str(auto_dir)])
    cli(
        [
            "render",
            str(sample_3d_nifti),
            "--min",
            "-5",
            "--max",
            "5",
            "--output-dir",
            str(manual_dir),
        ]
    )

    base = _base_name(sample_3d_nifti)
    auto = np.array(Image.open(auto_dir / f"{base}_axial.png"))
    manual = np.array(Image.open(manual_dir / f"{base}_axial.png"))
    assert int(auto.max()) - int(auto.min()) > int(manual.max()) - int(manual.min())


def test_min_greater_than_max_is_rejected(sample_3d_nifti, tmp_path, capsys):
    exit_code = cli(
        [
            "render",
            str(sample_3d_nifti),
            "--min",
            "5",
            "--max",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code != 0
    assert "Minimum intensity must be less than maximum intensity" in capsys.readouterr().err
