# https://openslide.org/formats/mirax/

import dask
import dask.array as da
import numpy as np
from ome_zarr import dask_utils
import openslide
import os.path
import skimage.transform as sk_transform

from src.ImageSource import ImageSource
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
        self.widths = [size[0] for size in self.slide.level_dimensions]
        self.heights = [size[1] for size in self.slide.level_dimensions]
        self.width, self.height = self.widths[0], self.heights[0]
        self.scales = [1 / downsample for downsample in self.slide.level_downsamples]
        self.nchannels = 4      # Mirax is RGBA
        self.source_shape = self.height, self.width, self.nchannels
        self.source_dim_order = 'yxc'
        self.name = get_filetitle(self.uri)
        nbits = 8
        for key, value in self.metadata.items():
            if 'slide_name' in key:
                self.name = value
            if 'slide_bitdepth' in key:
                nbits = int(value)
        self.dtype = np.dtype(f'uint{nbits}')

        self.shape = 1, self.nchannels, 1, self.height, self.width
        self.dim_order = 'tczyx'

        # OpenSlide stores microns per pixel in properties
        mpp_x = float(self.metadata.get(openslide.PROPERTY_NAME_MPP_X, 0))
        mpp_y = float(self.metadata.get(openslide.PROPERTY_NAME_MPP_Y, 0))
        self.pixel_size = {'x': mpp_x, 'y': mpp_y}


    def is_screen(self):
        # Mirax files are not multi-well screens
        return False

    def get_shape(self):
        return self.shape

    def get_data(self, well_id=None, field_id=None, as_dask=False, as_dask_pyramid=False, as_generator=False, **kwargs):
        """
        Gets image data.

        Returns:
            ndarray: Image data.
        """

        def read_tile_array(x, y, width, height, level=0):
            #print(f'reading {level}: ', y * width + x)
            return np.array(self.slide.read_region((x, y), level, (width, height)))

        def get_lazy_tile(x, y, width, height, level=0):
            lazy_array = dask.delayed(read_tile_array)(x, y, width, height, level)
            return da.from_delayed(lazy_array, shape=(height, width, self.nchannels), dtype=self.dtype)

        shape2 = self.source_shape[:2]
        if as_dask:
            y_chunks, x_chunks = da.core.normalize_chunks(TILE_SIZE, shape2, dtype=self.dtype)
            rows = []
            y = 0
            for height in y_chunks:
                row = []
                x = 0
                for width in x_chunks:
                    row.append(get_lazy_tile(x, y, width, height))
                    x += width
                rows.append(da.concatenate(row, axis=1))
                y += height
            data = da.concatenate(rows, axis=0)
            return redimension_data(data, self.source_dim_order, self.dim_order)
        elif as_dask_pyramid:
            pyramid = []
            scale = 1
            for level in range(PYRAMID_LEVELS + 1):
                shape = np.multiply(shape2, scale).astype(int)
                level, rescale = get_level_from_scale(self.scales, scale)
                y_chunks, x_chunks = da.core.normalize_chunks(TILE_SIZE, list(shape), dtype=self.dtype)
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
                if rescale != 1:
                    shape3 = tuple(list(shape) + [self.nchannels])
                    data = dask_utils.resize(data, shape3)
                data = redimension_data(data, self.source_dim_order, self.dim_order)
                pyramid.append(data)
                scale /= PYRAMID_DOWNSCALE
            return pyramid
        elif as_generator:
            def tile_generator(scale=1):
                level, rescale = get_level_from_scale(self.scales, scale)
                read_size = int(TILE_SIZE / rescale)
                for y in range(0, self.heights[level], read_size):
                    for x in range(0, self.widths[level], read_size):
                        data = np.array(self.slide.read_region((x, y), level, (read_size, read_size)))
                        if rescale != 1:
                            shape = np.multiply(shape2, scale).astype(int)
                            data = sk_transform.resize(data, shape, preserve_range=True).astype(data.dtype)
                        yield data
            return tile_generator
        else:
            data = read_tile_array(0, 0, self.width, self.height, level=0)
            return redimension_data(data, self.source_dim_order, self.dim_order)

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
            {"name": "Alpha", "color": [1, 1, 1, 1]}
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
        shape = self.get_shape()
        return np.prod(shape) * np.dtype(self.get_dtype()).itemsize

    def close(self):
        self.slide.close()
