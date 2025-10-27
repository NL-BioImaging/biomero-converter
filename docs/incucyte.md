# Incucyte Converter

## Overview

The Incucyte converter handles `.icarch` files from Incucyte live-cell imaging systems. These archives can contain data from multiple plates in a hierarchical directory structure.

## Directory Structure of Incucyte archive

```
*.icarch
├── EssenFiles
    └── ScanData
        └── YYMM (year/month)
            └── DD (day)
                └── HHMM (hour/minute)
                    ├── PlateID (e.g., "100", "101")
                    │   └── Well(A1|A2|..)-Position(1|2|..)-Channel(C1|C2|Ph).tif
                    └── Diag.zip
                        └── Diag.log (metadata including pixel sizes)
```

## Key Features

- **Multi-plate support**: A single `.icarch` file can contain multiple plates identified by PlateID folders
- **Automatic plate detection**: If no plate is specified, the converter will:
  - Use the first plate if multiple exist (with a warning)
  - Automatically process the single plate if only one exists
- **Metadata extraction**: Reads pixel sizes, magnification, and exposure times from `Diag.log`

## Command Line Usage

### Local Python Usage

#### Basic Conversion (Auto-detect plate)

```bash
python main.py --inputfile "path/to/experiment.icarch" \
               --outputfolder "path/to/output" \
               --outputformat ometiff
```

#### Specify Plate ID (Incucyte-specific option)

Use the `--plateid` option to convert a specific plate from an archive:

```bash
python main.py --inputfile "path/to/experiment.icarch" \
               --outputfolder "path/to/output" \
               --outputformat ometiff \
               --plateid 100
```

#### Full Example with All Options

```bash
python main.py --inputfile "D:\Data\incucyte_test\experiment.icarch" \
               --outputfolder "D:\Data\incucyte_test\ometiff" \
               --outputformat ometiff \
               --plateid 100 \
               --altoutputfolder "D:\Data\backup\ometiff" \
               --show_progress \
               --verbose
```

### Docker Usage

#### Basic Conversion with Docker

```bash
docker run -v /path/to/data:/data image-db-to-ome \
    --inputfile /data/experiment.icarch \
    --outputfolder /data/output \
    --outputformat ometiff
```

#### Specify Plate ID with Docker

```bash
docker run -v /path/to/data:/data image-db-to-ome \
    --inputfile /data/experiment.icarch \
    --outputfolder /data/output \
    --outputformat ometiff \
    --plateid 100
```

#### Windows Example with Docker

```bash
docker run -v D:\Data\incucyte_test:/data image-db-to-ome \
    --inputfile /data/experiment.icarch \
    --outputfolder /data/ometiff \
    --outputformat ometiff \
    --plateid 100 \
    --show_progress \
    --verbose
```

#### Docker with Alternative Output Folder

```bash
docker run -v /path/to/input:/input \
           -v /path/to/output:/output \
           -v /path/to/backup:/backup \
           image-db-to-ome \
    --inputfile /input/experiment.icarch \
    --outputfolder /output \
    --outputformat ometiff \
    --plateid 100 \
    --altoutputfolder /backup
```

## Plate ID Option (`--plateid`)

**This is unique to the Incucyte converter.**

- **Purpose**: Selects a specific plate when multiple plates exist in the same archive
- **Format**: String (e.g., `"100"`, `"101"`, `"702"`)
- **Behavior**:
  - If omitted with a single plate: Automatically processes that plate
  - If omitted with multiple plates: Processes the first plate and shows a warning
  - If specified: Only processes the specified plate

### Discovering Available Plates

To see which plates are available in an archive before conversion:

```python
from src.IncucyteSource import IncucyteSource

plates = IncucyteSource.get_available_plates("path/to/experiment.icarch")
print(f"Available plates: {plates}")
# Output: Available plates: ['100', '101', '702']
```

## Common Scenarios

### Single Plate Archive

**Local:**
```bash
python main.py --inputfile "experiment.icarch" --outputfolder "output" --outputformat ometiff
```

**Docker:**
```bash
docker run -v /path/to/data:/data image-db-to-ome \
    --inputfile /data/experiment.icarch \
    --outputfolder /data/output \
    --outputformat ometiff
```

### Multi-Plate Archive - Process All Plates

**Local:**
```bash
# Run conversion once per plate
python main.py --inputfile "experiment.icarch" --outputfolder "output" --outputformat ometiff --plateid 100
python main.py --inputfile "experiment.icarch" --outputfolder "output" --outputformat ometiff --plateid 101
python main.py --inputfile "experiment.icarch" --outputfolder "output" --outputformat ometiff --plateid 702
```

**Docker:**
```bash
# Run conversion once per plate
docker run -v /path/to/data:/data image-db-to-ome \
    --inputfile /data/experiment.icarch --outputfolder /data/output --outputformat ometiff --plateid 100

docker run -v /path/to/data:/data image-db-to-ome \
    --inputfile /data/experiment.icarch --outputfolder /data/output --outputformat ometiff --plateid 101

docker run -v /path/to/data:/data image-db-to-ome \
    --inputfile /data/experiment.icarch --outputfolder /data/output --outputformat ometiff --plateid 702
```

## Output

The converter generates:
- Output filename format: `{experiment_name}_plate{PlateID}.ome.{format}`
- Example: `incucyte_test_plate100.ome.tiff`

## Supported Channels

- **C1**: Green fluorescence
- **C2**: Red fluorescence  
- **Ph** or **P**: Phase contrast

## Notes

- Missing images are automatically filled with black frames to maintain consistent dimensions
- Timepoints are sorted chronologically
- Pixel sizes are extracted from `Diag.log` when available for accurate spatial calibration
- When using Docker, ensure all paths are mapped correctly using `-v` volume mounts