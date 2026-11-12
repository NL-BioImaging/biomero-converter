from biomero_converter.version import get_version

VERSION = get_version()

OME_DIR = 'OME'
OME_FILE = 'METADATA.ome.xml'

TILE_SIZE = 1024

TIFF_COMPRESSION = 'LZW'
#TIFF_COMPRESSION = 'JPEGXR_NDPI'
#TIFF_COMPRESSIONARGS = 75

ZARR_CHUNK_SIZE = TILE_SIZE
ZARR_SHARD_MULTIPLIER = 10

PYRAMID_DOWNSCALE = 2
PYRAMID_LEVELS = 6

RETRY_ATTEMPTS = 3

# Default output format: 'omezarr04', 'omezarr05', 'omezarr06' (OME-Zarr version) or 'ometiff'.
# Callers (e.g. GUIs) can use this as their default to stay in sync with the converter.
DEFAULT_OUTPUT_FORMAT = 'omezarr05'
