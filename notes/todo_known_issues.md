# Known issues and TODO

## Known issues

### Writing OME-Zarr to a Windows network share (#10, fixed in v0.1.31)

On some SMB shares, renaming onto an existing file raises `PermissionError [WinError 5]` instead of
`FileExistsError`, which zarr's `LocalStore.set_if_not_exists` does not handle (still so in zarr 3.4.0).
`patch_zarr_windows_network_move()` in `src/ome_zarr_util.py` replaces zarr's private
`zarr.storage._local._safe_move` - a zarr update could silently stop it applying.
Only simulated locally; not yet confirmed on the reporter's `L:` share.

### Leica

- Tile scans with gaps between their tiles (negative overlap, e.g. `NegOverlapTilescan-2t-3pos`,
  `TestTileScan16bit`) are separate positions, not stitched: one is selected by `position` index (`--position`),
  else a ValueError lists them. Its stage position is the image position (OME-Zarr translation). ConvertLeica returns
  a single-image `.lif` for these instead. `FieldX`/`FieldY` can be a running index (all `FieldY` 0) rather than a grid.
- Unsupported dimensions (wavelength, rotation, loop, ...) use only their first index.
- Tile flip/swap (`FlipX`/`FlipY`/`SwapXY`) follows ConvertLeica, but has not been compared against a
  LAS X merged image of the same tile scan.

### DICOM

- A folder or DICOMDIR is searched recursively; each series is an image, selected by its SeriesInstanceUID
  (`image_uuid`, default the first series). A series with slices of several orientations (e.g. a reformat with its
  reference image) keeps only its largest group of one orientation and size, with a warning.
- Compressed pixel data pydicom has no plugin for (no gdcm / pylibjpeg installed) is decoded with imagecodecs
  (JPEG Lossless, JPEG-LS, JPEG 2000); other compressions still fail.
- Multi-frame frames are time points only if `FrameIncrementPointer` names FrameTime(Vector), else slices; the
  frame time is not written as a t scale. Without PixelSpacing (e.g. ultrasound) the pixel size defaults to 1 mm.

### Acquisition metadata

Mapping onto the imaging metadata model is done (and tracked) in imaging-metadata-converter.

- TIFF vendor metadata comes from `src/tiff_metadata.py`, a copy of napari-meta-tiff `_metadata.py`: keep in sync.
- Zeiss Fibics (800 MB) round-trip not checked after the change (ran out of memory).

### Conversion

- `convert()` retries every exception `RETRY_ATTEMPTS` times, also deterministic ones, and its error
  message reports `RETRY_ATTEMPTS` instead of the `max_attempts` used.
- MIRAX conversion is very slow (single-threaded `read_region` via dask): `sample4.mrxs` took 40+ min for
  one format. iSyntax takes ~9 min per OME-Zarr format.
- NumPy 2.5 deprecates setting an array's shape, which tifffile still does when reading (DeprecationWarning).

### Tests

- `test_convert` checks every output (each Leica image / Incucyte plate) against its source image: pixel size
  (exact float equality), pixel data (level 0; per well and field for screens), wells, and position (single images).
- `EM04573_01small.ome.tif` from the default test list is missing locally.
- tifffile < 2026.9.20 reports squeezed shape/axes for 'shaped' series (our OME-TIFF output) but returns
  unsqueezed data, so reading back our OME-TIFF gave a wrong shape; fixed by tifffile 2026.9.20 (pinned as minimum).

## In progress

test_convert pixel data and all outputs, OME-TIFF Plane positions for single images, OME-TIFF plate image refs keyed
by integer field id, also in OmeZarrSource (2026-10-01). Small varied files, DB/CellsSmall (all formats) and the
Incucyte archive (OME-TIFF) pass; Incucyte / DB OME-Zarr re-run after the OmeZarrSource field fix pending.

## TODO

- Report the SMB `PermissionError` case upstream to zarr-python (`set_if_not_exists`).
- Don't retry deterministic errors in `convert()`; fix the retry count in its message.
