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

With `--rows`, each panel is a render output folder, and each row takes that folder's `*_<pattern>.png`. A folder of `--frames-as-png` output gives one column per `*_<pattern>_vol-NNNN.png` frame instead, so frames and single-volume renders, or several sessions, can share one grid:

```sh
mrplot compose --output asl.png --rows axial,coronal,sagittal \
    --panel "TI 1.57 s,TI 2.46 s,TI 4.5 s" ses-01/delays \
    --panel Console ses-01/console --panel BASIL ses-01/basil
```

Give a frame folder one label per frame, separated by commas (`\,` for a literal comma), or a single label, which is numbered by volume index. `--frames 0,2` or `--frames 0-2` takes only some volumes. Each row must use the same number of columns from each folder. With `--rows`, `--columns` defaults to the width of a row and must match it if given.

The same is available from Python as `mrplot.compose.compose_files` and `panels_from_directories`.

## Tests

Run `uv run pytest`. The suite generates its own synthetic NIfTI phantoms.
