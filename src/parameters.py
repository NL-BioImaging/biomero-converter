VERSION = 'v0.1.19'

TILE_SIZE = 1024

TIFF_COMPRESSION = 'LZW'
#TIFF_COMPRESSION = 'JPEGXR_NDPI'
#TIFF_COMPRESSIONARGS = 75

ZARR_CHUNK_SIZE = TILE_SIZE
ZARR_SHARD_MULTIPLIER = 10

PYRAMID_DOWNSCALE = 2
PYRAMID_LEVELS = 0           # 0 = auto (based on PYRAMID_MIN_SIZE)
PYRAMID_MIN_SIZE = 256       # stop adding levels when smallest dim reaches this

MAX_WORKERS = 4              # thread pool workers for parallel well writing

RETRY_ATTEMPTS = 3
