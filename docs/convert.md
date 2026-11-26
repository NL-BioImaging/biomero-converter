# converter.convert() Usage

The `convert()` function in `biomero_converter/converter.py` is used to convert an input file to OME format and write the result to one or more output folders.

## Example Usage

```python
from biomero_converter.converter import convert

input_filename = "path/to/input_file"
output_folder = "path/to/output_folder"
alt_output_folder = "path/to/alternative_output_folder"

if __name__ == '__main__':
    result_json = convert(
        input_filename,
        output_folder,
        alt_output_folder,
        output_format="omezarr05",
        show_progress=False,
        verbose=False
    )

    print(result_json)
```

Run a script calling `convert()` under `if __name__ == '__main__':`: some formats (iSyntax) are read in parallel in
worker processes, which import the script.

Images larger than the available memory (on a Slurm node: the job's memory limit) are read lazily and written per
chunk or tile, using the dask threads (all CPUs of the job) for reading (where the format allows), compressing and
writing.

## Arguments

- `input_filename` (str): Path to the input file.
- `output_folder` (str): Output folder path.
- `alt_output_folder` (str, optional): Alternative output folder path.
- `output_format` (str): Output format: `'omezarr04'`, `'omezarr05'`, `'omezarr06'` (OME-Zarr version) or `'ometiff'` (default: `'omezarr05'`, available as `biomero_converter.parameters.DEFAULT_OUTPUT_FORMAT`; OME-Zarr input defaults to `'ometiff'`).
- `show_progress` (bool): If `True`, prints progress.
- `verbose` (bool): If `True`, enables verbose logging.
- `progress` (callable, optional): Called with the progress of the conversion as a fraction from 0 to 1, e.g. to
  show a progress bar in an application. Progress is reported per well and field for plates, per pyramid level
  and tile while writing images, and per finished dask task. Reading an image fully into memory is one step, so
  progress can pause there. A retried conversion starts again from 0. The command line does not use it.
- `overwrite` (bool, default `True`): If `False`, outputs that already exist are not written again; their result
  has `'skipped': True`. An output that would replace the input, or be written inside it, is always refused with
  `biomero_converter.converter.OutputPathError` (not retried).
- `**kwargs`: Source specific parameters, e.g.:
  - `plate_id` (str): Incucyte plate ID (default: all plates).
  - `image_uuid` (str) or `image_index` (int): Leica image to convert (default: all images in the file).
  - `crop_to_scan` (bool, default True): MIRAX: crop to the scanned region, leaving out the unscanned (background)
    area around it; no data lost, about twice as fast. The image position is the region's offset in the slide.
    False (command line: `--no-crop_to_scan`; GUI: "Crop slides to the scanned region") keeps the whole slide.

## Returns

A JSON string containing an array with conversion result info, e.g.:

```json
[
  {
    "name": "experiment_name",
    "full_path": "path/to/output_folder/experiment_name.ome.zarr",
    "alt_path": "path/to/alternative_output_folder/experiment_name.ome.zarr"
  }
]
```

