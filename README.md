# mrplot

Render axial, coronal, and sagittal views of NIfTI volumes. A 3D volume produces three PNGs. A 4D volume produces three MP4s, or one PNG per volume and plane with `--frames-as-png`.

## Quick start

```sh
uv sync
uv run mrplot render scan.nii.gz --output-dir out
```

Run `uv run mrplot render --help` for CLI options. See the [parcellations and overlays guide](docs/overlays.md) for lookup tables, ROI outlines, image stacking, and Python examples.

## Tests

Run `uv run pytest`. The suite generates its own synthetic NIfTI phantoms.
