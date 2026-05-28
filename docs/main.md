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
    --outputformat omezarr2 \
    --show_progress \
    --verbose
```

Replace `/local/input_folder` and `/local/output_folder` with your actual local paths.  
Adjust the image name (`biomero-converter:latest`) as needed.

## Arguments

Refer to the main script for all available arguments:

- `--inputfile`: Path to the input file (required)
- `--outputfolder`: Path to the output folder (required)
- `--altoutputfolder`: Alternative output folder (optional)
- `--outputformat`: Output format version (default: `omezarr2`)
- `--show_progress`: Show progress bar (flag)
- `--verbose`: Enable verbose logging (flag)

## Alternatively Run the main script directly

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main script with the required arguments.

```sh
python main.py \
    --inputfile /data/input/input_file.tiff \
    --outputfolder /data/output \
    --outputformat omezarr2 \
    --show_progress \
    --verbose
    
usage: main.py [-h] --inputfile INPUTFILE --outputfolder OUTPUTFOLDER [--altoutputfolder ALTOUTPUTFOLDER] [--outputformat OUTPUTFORMAT] [--show_progress] [--verbose] [--plateid PLATEID]
main.py: error: the following arguments are required: --inputfile, --outputfolder
```
