# mrplot

Render axial, coronal, and sagittal views of NIfTI volumes. A 3D volume produces three PNGs. A 4D volume produces three MP4s, or one PNG per volume and plane with `--frames-as-png`.

## Quick start

```sh
uv sync
uv run mrplot render scan.nii.gz --output-dir out
```

Run `uv run mrplot render --help` for CLI options. See the [parcellations and overlays guide](docs/overlays.md) for lookup tables, ROI outlines, image stacking, and Python examples.

## Titles

Each plane is titled with the base image and the layers drawn on it, e.g. `T1 + seg (outline) Axial Slice`. Use `--title` to set it (with optional `{name}`, `{layers}` and `{plane}` fields), or `--no-title` to leave it off.

## Composing figures

`mrplot compose` tiles rendered PNGs into one figure, with optional labels under each image and a title above:

```sh
mrplot compose --output overview.png --columns 2 --title sub-01 \
    --panel T1 t1/scan_axial.png --panel Seg seg/scan_axial.png
mrplot compose --output grid.png --rows axial,coronal,sagittal --panel T1 t1 --panel Seg seg
```

With `--rows`, each panel is a render output folder, and each row takes that folder's `*_<pattern>.png`. The same is available from Python as `mrplot.compose.compose_files`.

## Tests

Run `uv run pytest`. The suite generates its own synthetic NIfTI phantoms.
