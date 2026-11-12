# biomero-converter <img src="docs/images/logo.png" alt="biomero-converter logo" width="64" align="right">

Conversion from various formats to Ome-tiff and Ome-zarr

This tool can be used as biomero plugin, Docker container, stand-alone command line tool, Python package or desktop application (GUI)

![biomero-converter desktop GUI](docs/images/gui.png)

## Features
- Extract and process image data including bioformats unsupported formats
  - OME-Zarr
  - (OME-) Tiff
  - Thermo Fisher tiff files
  - ImageXpress Pico
  - Molecular Devices
  - CellReporterXpress experiment
  - image db files (.db)
  - isyntax files (.isyntax)
  - Incucyte archives (.icharch)
  - 3DHistech files (.mrxs)
  - DICOM files
  - Leica files (.lif, .lof, .xlef, .xlcf, .xlif)
    - all images in a file, or a single image selected by UUID
    - RGB, multi-channel, Z-stack, time series and tile scans (stitched)
- Export to Ome formats
  - OME-Tiff
  - OME-Zarr 0.4, 0.5 (default) and 0.6
- Supporting
  - (Large) whole slide images
  - Screen Plate Well / High Content Screening format
- Compatible with Omero
- Desktop GUI for Windows, macOS and Linux (see below)

## Desktop GUI
Convert files without the command line, on Windows, macOS and Linux (screenshot above):

1. Install the package with the GUI extra (Python 3.12 or newer):
   ```sh
   pip install "biomero-converter[gui]"
   ```
2. Start the application:
   ```sh
   biomero-converter-gui
   ```
   (or `python -m biomero_converter.gui.app`)
3. **Add files** or **Add folder**, and choose next to each input what to convert: all or one of its images (Leica), plates (Incucyte) or series (DICOM). **Browse** to an output folder (or enter a relative path, e.g. `converted`, to write next to each input), choose the output format (default OME-Zarr 0.5) and click **Convert**.

The GUI uses Tk, which comes with Python on Windows (python.org or conda). On macOS use Python from python.org or conda; on Linux install your distribution's Tk package (e.g. `sudo apt install python3-tk`) or use conda.
See the [Desktop GUI documentation](https://nl-bioimaging.github.io/biomero-converter/gui/) for all options.

## Documentation
### [nl-bioimaging.github.io/biomero-converter](https://nl-bioimaging.github.io/biomero-converter/)
### Generated: [deepwiki.com/NL-BioImaging/biomero-converter](https://deepwiki.com/NL-BioImaging/biomero-converter)

## Use of AI

Though completely human-designed and created, generative AI coding assistants were used in developing this software, including tests and documentation.
All changes are directed and approved by a maintainer, and checked by automated tests where applicable.

## License

[Apache-2.0]


[Apache-2.0]: http://opensource.org/licenses/Apache-2.0
