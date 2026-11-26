# PyInstaller hook for biomero-converter.
#
# biomero-converter and several of its dependencies load data files and native libraries
# at runtime, which PyInstaller's import analysis does not find by itself:
# - sources are imported inside functions, per input format (helper.create_source)
# - OpenSlide / iSyntax / codecs ship native libraries
# - imaging-metadata-converter loads its model index, units and mappings (.json, .yaml) at
#   runtime; without them conversion fails with a missing file. It needs no LinkML to map.
from PyInstaller.utils.hooks import collect_all

# import names (not pip names: pyisyntax is imported as 'isyntax')
PACKAGES = (
    'biomero_converter',
    'openslide', 'openslide_bin', 'isyntax', 'imagecodecs', 'pydicom', 'liffile',
    'ome_zarr', 'zarr', 'numcodecs', 'rocrate',
    'imaging_metadata_converter',
)

datas, binaries, hiddenimports = [], [], []
for _package in PACKAGES:
    _datas, _binaries, _hiddenimports = collect_all(_package)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hiddenimports
