# mrplot

Renders orthogonal (axial/coronal/sagittal) plots or videos from a single NIfTI file.

## Install

```
uv sync
```

## Usage

```
mrplot render NIFTI_FILE --output-dir DIR [options]
```

3D input -> one PNG per plane. 4D input -> one MP4 per plane (one frame per timepoint). 5D and other dimensionalities are rejected.

Output files are named `<input_basename>_<plane>.png` (or `.mp4`), one per plane (`axial`, `coronal`, `sagittal`), written into `--output-dir`. On success, the written file paths are printed to stdout, one per line. On error, a message is printed to stderr and the exit code is non-zero.