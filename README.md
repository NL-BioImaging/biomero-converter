# biomero-converter

Conversion from various formats to Ome-tiff and Ome-zarr

This tool can be used as biomero plugin, Docker container or stand-alone

![logo](docs/images/logo.png)

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

## Documentation
### [nl-bioimaging.github.io/biomero-converter](https://nl-bioimaging.github.io/biomero-converter/)
### Generated: [deepwiki.com/NL-BioImaging/biomero-converter](https://deepwiki.com/NL-BioImaging/biomero-converter)

## Use of AI

Though completely human-designed and created, generative AI coding assistants were used in developing this software, including tests and documentation.
All changes are directed and approved by a maintainer, and checked by automated tests where applicable.

## License

[Apache-2.0]


[Apache-2.0]: http://opensource.org/licenses/Apache-2.0
