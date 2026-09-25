# Known issues and TODO

## Known issues

### Writing OME-Zarr to a Windows network share (#10, fixed in v0.1.31)

On some SMB shares, renaming onto an existing file raises `PermissionError [WinError 5]` instead of
`FileExistsError`, which zarr's `LocalStore.set_if_not_exists` does not handle (still so in zarr 3.4.0).
`patch_zarr_windows_network_move()` in `src/ome_zarr_util.py` replaces zarr's private
`zarr.storage._local._safe_move` - a zarr update could silently stop it applying.
Only simulated locally; not yet confirmed on the reporter's `L:` share.

### Leica

- Tile scans with negative overlap (gaps between tiles, e.g. `NegOverlapTilescan-2t-3pos`) are stitched
  edge to edge with a warning. ConvertLeica returns a single-image `.lif` for these instead.
- Confocal channel names stay generic (`Ch0`, ...): their names/dyes are spread over sequential scan
  settings. Widefield channels get names and emission from `WideFieldChannelInfo`.
- Unsupported dimensions (wavelength, rotation, loop, ...) use only their first index.
- Tile flip/swap (`FlipX`/`FlipY`/`SwapXY`) follows ConvertLeica, but has not been compared against a
  LAS X merged image of the same tile scan.

### Conversion

- `convert()` retries every exception `RETRY_ATTEMPTS` times, also deterministic ones, and its error
  message reports `RETRY_ATTEMPTS` instead of the `max_attempts` used.
- MIRAX conversion is very slow (single-threaded `read_region` via dask): `sample4.mrxs` took 40+ min for
  one format. iSyntax takes ~9 min per OME-Zarr format.
- NumPy 2.5 deprecates setting an array's shape, which tifffile still does when reading (DeprecationWarning).

### Tests

- `test_convert` compares only pixel size (exact float equality) and wells, not pixel data, and only the
  first output of multi-image (e.g. Leica) files.
- `EM04573_01small.ome.tif` from the default test list is missing locally.
- tifffile < 2026.9.20 reports squeezed shape/axes for 'shaped' series (our OME-TIFF output) but returns
  unsqueezed data, so reading back our OME-TIFF gave a wrong shape; fixed by tifffile 2026.9.20 (pinned as minimum).
- Leica: old LAS AF files (e.g. SP5) have `HardwareSettingList` with flat `ScannerSetting`/`FilterSetting`
  record lists instead of `HardwareSetting`; not included in acquisition metadata yet.

## In progress

(none)

## TODO

- Report the SMB `PermissionError` case upstream to zarr-python (`set_if_not_exists`).
- Don't retry deterministic errors in `convert()`; fix the retry count in its message.
- Leica: confocal channel names / excitation / emission; option for negative overlap tile scans.
- Compare pixel data in `test_convert`, and check all outputs of multi-image files.
