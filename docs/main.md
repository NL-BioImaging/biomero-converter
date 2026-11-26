# Run the Main Script

## Using Docker

You can use Docker to run the conversion process by mounting your input and output folders and passing the required arguments.

## Example

```sh
docker run --rm \
  -v /local/input_folder:/data/input \
  -v /local/output_folder:/data/output \
  cellularimagingcf/biomero-converter:latest \
    --inputfile /data/input/input_file.tiff \
    --outputfolder /data/output \
    --outputformat omezarr05 \
    --show_progress \
    --verbose
```

Replace `/local/input_folder` and `/local/output_folder` with your actual local paths.  
Adjust the image name (`biomero-converter:latest`) as needed.

## Arguments

Refer to the main script for all available arguments:

- `--inputfile`: Path to the input file (required)
- `--outputfolder`: Path to the output folder (required to convert)
- `--metadata [VIEW]`: Only print the input's metadata as YAML, without converting (no output folder needed).
  `VIEW` is `source` (the vendor metadata as read), `model` (mapped onto the imaging metadata model), `spatial`
  (the pixel size, position and rotation the conversion writes) or `all` (default). Combine with `--image_uuid` or
  `--plateid` for one part of the input.
- `--altoutputfolder`: Alternative output folder (optional)
- `--outputformat`: Output format: `omezarr04`, `omezarr05`, `omezarr06` (OME-Zarr version) or `ometiff` (default: `omezarr05`)
- `--show_progress`: Show progress bar (flag)
- `--verbose`: Enable verbose logging (flag)
- `--plateid`: Incucyte plate ID (optional, default: all plates)
- `--image_uuid`: Leica image UUID (optional, default: all images in the file)

## Alternatively install the Python package

1. Install the package from PyPI using `pip install biomero-converter`
   (or, from a clone of the repository, `pip install -e .`).
2. Run the `biomero-converter` command with the required arguments.

```sh
biomero-converter \
    --inputfile /data/input/input_file.tiff \
    --outputfolder /data/output \
    --outputformat omezarr05 \
    --show_progress \
    --verbose
    
usage: biomero-converter [-h] --inputfile INPUTFILE [--outputfolder OUTPUTFOLDER] [--metadata [{all,source,model,spatial}]] ...
biomero-converter: error: --outputfolder is required to convert (or use --metadata to only print the metadata)
```

To check what is read from a file before converting it:

```sh
biomero-converter --inputfile /data/input/input_file.tiff --metadata spatial
```

## Desktop GUI

To convert files without the command line, install the GUI extra and start the desktop application:

```sh
pip install "biomero-converter[gui]"
biomero-converter-gui
```

See [Desktop GUI](gui.md).
