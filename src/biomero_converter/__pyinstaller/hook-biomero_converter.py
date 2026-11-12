# PyInstaller hook for biomero-converter.
#
# biomero-converter and several of its dependencies load data files and native libraries
# at runtime, which PyInstaller's import analysis does not find by itself:
# - sources are imported inside functions, per input format (helper.create_source)
# - OpenSlide / iSyntax / codecs ship native libraries
# - imaging-metadata-converter loads LinkML schemas (.yaml), mappings (.json) and prefix
#   registries at runtime; without them conversion fails (first with a missing file, then
#   with a misleading circular-import error from linkml_runtime on retry)
from PyInstaller.utils.hooks import collect_all

# import names (not pip names: pyisyntax is imported as 'isyntax')
PACKAGES = (
    'biomero_converter',
    'openslide', 'openslide_bin', 'isyntax', 'imagecodecs', 'pydicom', 'liffile',
    'ome_zarr', 'zarr', 'numcodecs', 'rocrate',
    'imaging_metadata_converter', 'linkml_runtime', 'prefixmaps', 'prefixcommons',
    'curies', 'jsonschema_specifications',
)

datas, binaries, hiddenimports = [], [], []
for _package in PACKAGES:
    _datas, _binaries, _hiddenimports = collect_all(_package)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hiddenimports
