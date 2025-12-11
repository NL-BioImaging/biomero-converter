# https://openslide.org/formats/mirax/

import dask
import dask.array as da
import numpy as np
import openslide
import skimage.transform as sk_transform

from src.ImageSource import ImageSource
from src.color_conversion import hexrgb_to_rgba
from src.parameters import *
from src.util import redimension_data, get_level_from_scale, get_filetitle


class MiraxSource(ImageSource):
    """
    ImageSource subclass for reading Mirax files using OpenSlide.
    """

    def __init__(self, uri, metadata={}):
        super().__init__(uri, metadata)
        self.slide = openslide.open_slide(uri)

    def init_metadata(self):
        self.metadata = {key.lower(): value for key, value in dict(self.slide.properties).items()}

        self.dimensions = self.slide.level_dimensions
        self.widths = [width for width, height in self.slide.level_dimensions]
        self.heights = [height for width, height in self.slide.level_dimensions]
        self.level_downsamples = self.slide.level_downsamples
        self.scales = [1 / downsample for downsample in self.level_downsamples]
        self.nchannels = 3      # Mirax is RGBA; convert to RGB
        self.shapes = [(height, width, self.nchannels) for (width, height) in self.dimensions]
        self.shape = self.shapes[0]
        self.dim_order = 'yxc'
        self.is_rgb_channels = True
        nbits = 8
        for key, value in self.metadata.items():
            if 'slide_name' in key:
                self.name = value
            if 'slide_bitdepth' in key:
                nbits = int(value)
        self.dtype = np.dtype(f'uint{nbits}')

        # OpenSlide stores microns per pixel in properties
        mpp_x = float(self.metadata.get(openslide.PROPERTY_NAME_MPP_X, 1))
        mpp_y = float(self.metadata.get(openslide.PROPERTY_NAME_MPP_Y, 1))
        self.pixel_size = {'x': mpp_x, 'y': mpp_y}
        background_float = hexrgb_to_rgba(self.metadata.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR, '000000'))[:3]
        self.background = [np.uint8(value * 255) for value in background_float]

        self.name = get_filetitle(self.uri)
        return self.metadata

    def is_screen(self):
        # Mirax files are not multi-well screens
        return False

    def get_shape(self):
        return self.shape

    # TODO: check (x/y) source data is read in order first to last (currently last to first) using dask, or use generator/stream to dask?
    # read_tile_array(50000, 180000, 1000, 1000, 0)

    def read_array(self, x, y, width, height, level=0):
        # OpenSlide uses (x, y) coordinates in level 0 reference size
        x0 = int(x * self.level_downsamples[level])
        y0 = int(y * self.level_downsamples[level])
        #return np.array(self.slide.read_region((x0, y0), level, (width, height)).convert('RGB'))   # discard alpha
        rgba = np.array(self.slide.read_region((x0, y0), level, (width, height)))
        alpha = np.atleast_3d(rgba[..., 3] / np.float32(255))
        rgb = (rgba[..., :3] * alpha + self.background * (1 - alpha)).astype(np.uint8)
        return rgb

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        data = self.read_array(0, 0, self.widths[level], self.heights[level], level=level)
        return redimension_data(data, self.dim_order, dim_order)

    def get_data_as_dask(self, dim_order, level=0, **kwargs):
        dask.config.set(scheduler='single-threaded')

        def get_lazy_tile(x, y, width, height, level=0):
            lazy_array = dask.delayed(self.read_array)(x, y, width, height, level)
            return da.from_delayed(lazy_array, shape=(height, width, self.nchannels), dtype=self.dtype)

        y_chunks, x_chunks = da.core.normalize_chunks(TILE_SIZE, self.shapes[level][:2], dtype=self.dtype)
        rows = []
        y = 0
        for height in y_chunks:
            row = []
            x = 0
            for width in x_chunks:
                row.append(get_lazy_tile(x, y, width, height, level=level))
                x += width
            rows.append(da.concatenate(row, axis=1))
            y += height
        data = da.concatenate(rows, axis=0)
        return redimension_data(data, self.dim_order, dim_order)

    def get_data_as_generator(self, dim_order, **kwargs):
        def data_generator(scale=1):
            level, rescale = get_level_from_scale(self.scales, scale)
            read_size = int(TILE_SIZE / rescale)
            for y in range(0, self.heights[level], read_size):
                for x in range(0, self.widths[level], read_size):
                    data = self.read_array(x, y, read_size, read_size, level)
                    if rescale != 1:
                        shape = np.multiply(data.shape[:2], rescale).astype(int)
                        data = sk_transform.resize(data, shape, preserve_range=True).astype(data.dtype)
                    yield redimension_data(data, self.dim_order, dim_order)
        return data_generator

    def get_name(self):
        return self.name

    def get_dim_order(self):
        return self.dim_order

    def get_dtype(self):
        return self.dtype

    def get_pixel_size_um(self):
        return self.pixel_size

    def get_position_um(self, well_id=None):
        # Not applicable for Mirax
        return {'x': 0, 'y': 0}

    def get_channels(self):
        # Mirax is RGB, return NGFF-style channel metadata
        return [
            {"name": "Red", "color": [1, 0, 0, 1]},
            {"name": "Green", "color": [0, 1, 0, 1]},
            {"name": "Blue", "color": [0, 0, 1, 1]},
            #{"name": "Alpha", "color": [1, 1, 1, 1]}
        ]

    def get_nchannels(self):
        return self.nchannels

    def is_rgb(self):
        return True

    def get_rows(self):
        return []

    def get_columns(self):
        return []

    def get_wells(self):
        return []

    def get_time_points(self):
        return []

    def get_fields(self):
        return []

    def get_acquisitions(self):
        return []

    def get_total_data_size(self):
        return np.prod(self.shape) * np.dtype(self.get_dtype()).itemsize

    def close(self):
        self.slide.close()
