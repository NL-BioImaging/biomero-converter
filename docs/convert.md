# converter.convert() Usage

The `convert()` function in `converter.py` is used to convert an input file to OME format and write the result to one or more output folders.

## Example Usage

```python
from converter import convert

input_filename = "path/to/input_file"
output_folder = "path/to/output_folder"
alt_output_folder = "path/to/alternative_output_folder"

result_json = convert(
    input_filename,
    output_folder,
    alt_output_folder,
    output_format="omezarr2",
    show_progress=False,
    verbose=False
)

print(result_json)
```

## Arguments

- `input_filename` (str): Path to the input file.
- `output_folder` (str): Output folder path.
- `alt_output_folder` (str, optional): Alternative output folder path.
- `output_format` (str): Output format string (default: `'omezarr2'`).
- `show_progress` (bool): If `True`, prints progress.
- `verbose` (bool): If `True`, enables verbose logging.

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

