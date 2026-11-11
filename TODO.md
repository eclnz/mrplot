# TODO

- Manual slice index selection (`--sagittal-index`/`--coronal-index`/`--axial-index`,
  or similar) so a slice can be picked directly without building a mask file.
  Worth generalizing to physical position (mm) rather than voxel index, since
  that's what fractional positions naturally come from.
- Through-plane cubic (Catmull-Rom) reconstruction: extract the 4 neighboring
  slices along the fixed axis and interpolate across them, instead of taking
  the single nearest voxel slice. Only meaningful once slice selection
  supports fractional/physical positions (see above) - at an integer index
  it's a no-op. Decide whether this is driven by the existing
  `--interpolation` flag or a separate one (e.g. `--depth-interpolation`).
- `--plane {axial,coronal,sagittal}` (repeatable) to render only the
  requested plane(s) instead of always all three.
- `--timepoint N` for 4D input: render one frame as a PNG instead of the
  whole video.
- `--percentile LOW HIGH` (e.g. `1 99`) as an alternative auto-window to raw
  min/max, more robust to outlier/noise voxels. Complements `--min`/`--max`.
- `--version` flag.
- Apply the "read full fast-axis, slice in memory" fix for sagittal-plane
  extraction (see scripts/benchmark.py findings) only when a crop window
  bounds the read, to speed up the masked/cropped path without sacrificing
  laziness on the default uncropped view.
