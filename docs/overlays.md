# Parcellations and overlays

## Parcellation colours

Use `--parcellation tableau` or `--parcellation vivid` to colour a volume of integer labels. The scheme assigns colours in ascending numeric label order across the whole image, including all frames. Label `0` is background by default. Use `--parcellation-background LABEL` to select a different label. Categorical rendering uses nearest-neighbour resampling and the plot background colour for background voxels. The intensity window and colorbar apply to continuous images.

Use `--parcellation-lut PATH` to use an atlas lookup table. mrplot reads FreeSurfer LUTs, FSLeyes `.lut` files, and text tables with one `LABEL #RRGGBB` pair per line. For FreeSurfer, mrplot uses the RGB channels. For FSLeyes, it converts channels from 0–1 to 8-bit RGB. Blank lines and full-line comments starting with `#` are allowed. An image label missing from a supplied table is an error unless it is the configured background. The LUT may include labels absent from the image. [FreeSurfer colour table](https://github.com/freesurfer/freesurfer/blob/dev/distribution/FreeSurferColorLUT.txt) · [FSLeyes LUT format](https://pages.fmrib.ox.ac.uk/fsl/fsleyes/fsleyes/userdoc/customising.html#lookup-tables)

```sh
mrplot render aparc+aseg.nii.gz --output-dir out \
  --parcellation-lut "$FREESURFER_HOME/FreeSurferColorLUT.txt"
```

## Outline overlays

Use `--outline` on a continuous scan or a parcellation. The outline image must contain finite integer labels on the same spatial voxel grid as the base. Equivalent axis flips and permutations are handled automatically. A 3D overlay can be reused across a 4D scan, and a 4D overlay must have the same number of frames as the base. The 500-value limit counts every distinct label in the outline image, including background if present. The background label defaults to `0` and can be selected with `--outline-background LABEL`.

Outlines follow the inside edge of each displayed region, including both sides of shared borders. The renderer checks neighbouring labels beyond the crop to locate region boundaries. Background has no outline colour. Use `--outline-scheme vivid` or `--outline-lut PATH` to reuse the categorical schemes or a lookup table. For exact positional colours, use `--outline-colors` or `--outline-colors-file`. Each colour corresponds to one foreground label in ascending numeric order, including labels absent from the current slice or frame.

### Runnable CLI and Python examples

Generate a small anatomy volume, mask, and parcellation with foreground labels `3`, `7`, and `42`.

```sh
python - <<'PY'
import nibabel as nib
import numpy as np

labels = np.zeros((24, 24, 24), dtype=np.int16)
labels[4:12, 4:20, 4:20] = 3
labels[12:17, 4:20, 4:20] = 7
labels[17:20, 4:20, 4:20] = 42
anatomy = np.indices(labels.shape, dtype=np.float32).sum(axis=0)
for name, voxels in [('anatomy', anatomy), ('mask', (labels > 0).astype(np.uint8)), ('parcels', labels)]:
    nib.save(nib.Nifti1Image(voxels, np.eye(4)), f'{name}.nii.gz')
PY

mrplot render parcels.nii.gz --output-dir out/fill --parcellation vivid
mrplot render anatomy.nii.gz --output-dir out/mask \
  --outline mask.nii.gz --outline-colors '#FF8800'
mrplot render anatomy.nii.gz --output-dir out/inline \
  --outline parcels.nii.gz --outline-colors '#E41A1C' '#377EB8' '#4DAF4A'

# One hex colour per line, for labels 3, 7 and 42 respectively.
cat > outlines.txt <<'EOF'
#E41A1C
#377EB8
#4DAF4A
EOF
mrplot render anatomy.nii.gz --output-dir out/from-file \
  --outline parcels.nii.gz --outline-colors-file outlines.txt
```

```python
from mrplot.categorical import CategoricalColors
from mrplot.plotting import render_file
from mrplot.rendering import RenderConfig

atlas_colors = CategoricalColors(mapping={3: "#E41A1C", 7: "#377EB8", 42: "#4DAF4A"})
render_file("parcels.nii.gz", "out/python-fill", config=RenderConfig(parcellation=atlas_colors))
render_file("anatomy.nii.gz", "out/python-mask", outline_path="mask.nii.gz",
            outline_colors=CategoricalColors(colors=["#FF8800"]))
render_file("anatomy.nii.gz", "out/python-lut", outline_path="parcels.nii.gz",
            outline_colors=atlas_colors)
render_file("anatomy.nii.gz", "out/python-file", outline_path="parcels.nii.gz",
            outline_colors=CategoricalColors.from_color_file("outlines.txt"))
```

For a native atlas lookup table, pass `CategoricalColors.from_lookup_table(path)` as `parcellation` in `RenderConfig` or as `outline_colors` in `render_file`.

## Layering images and ROIs

Repeat `--overlay IMAGE` for continuous images and `--outline LABELS` for masks or parcellations. Layers are drawn in the order their paths appear, with later layers on top. Put each layer's options after its path. Every layer uses the same grid and frame alignment rules as an outline. The `--mask` option still selects slices and controls cropping.

- `--overlay-opacity` sets image opacity (default `0.5`). Zero and nonfinite voxels are transparent before resampling. Each image has its own `--overlay-color-scale` (default `heat`), `--overlay-min`, and `--overlay-max`. Automatic bounds use the finite values in the displayed slices across all frames.
- `--outline-fill-opacity` (also `--mask-opacity`) sets ROI fill opacity (default `0`). `--outline-opacity` sets line opacity (default `1`). Both use the layer's label colours. At a boundary the line blends over the fill.
- Opacity values range from `0` for transparent to `1` for opaque.

```sh
mrplot render anatomy.nii.gz --output-dir out/layers \
  --overlay scan-a.nii.gz --overlay-opacity 0.35 \
  --overlay scan-b.nii.gz --overlay-opacity 0.25 --overlay-color-scale blue-red \
  --outline parcels.nii.gz --outline-lut atlas.txt --outline-opacity 0.8 \
  --outline mask.nii.gz --outline-colors '#00E5FF' --mask-opacity 0.15 --outline-opacity 0.6
```

```python
from mrplot.categorical import CategoricalColors
from mrplot.colors import BLUE_RED
from mrplot.overlays import ImageOverlay, LabelOverlay
from mrplot.plotting import render_file

render_file("anatomy.nii.gz", "out/layers", overlays=[
    ImageOverlay("scan-a.nii.gz", opacity=0.35),
    ImageOverlay("scan-b.nii.gz", opacity=0.25, color_scale=BLUE_RED),
    LabelOverlay("parcels.nii.gz", colors=CategoricalColors.from_lookup_table("atlas.txt"),
                 line_opacity=0.8),
    LabelOverlay("mask.nii.gz", colors=CategoricalColors(colors=["#00E5FF"]),
                 fill_opacity=0.15, line_opacity=0.6),
])
```
