# Uses https://github.com/anibali/pyisyntax
# which is based on https://github.com/amspath/libisyntax


import dask
import dask.array as da
from isyntax import ISyntax
import numpy as np
import skimage.transform as sk_transform
from xml.etree import ElementTree

from src.ImageSource import ImageSource
from src.parameters import *
from src.util import get_filetitle, xml_content_to_dict, redimension_data, get_level_from_scale


class ISyntaxSource(ImageSource):
    """
    Loads image and metadata from ISyntax format files.
    """
    def init_metadata(self):
        # read XML metadata header
        data = b''
        block_size = 1024 * 1024
        end_char = b'\x04'   # EOT character
        with open(self.uri, mode='rb') as file:
            done = False
            while not done:
                data_block = file.read(block_size)
                if end_char in data_block:
                    index = data_block.index(end_char)
                    data_block = data_block[:index]
                    done = True
                data += data_block

        self.metadata = xml_content_to_dict(ElementTree.XML(data.decode()))
        if 'DPUfsImport' in self.metadata:
            self.metadata = self.metadata['DPUfsImport']

        image = None
        image_type = ''
        for image0 in self.metadata.get('PIM_DP_SCANNED_IMAGES', []):
            image = image0.get('DPScannedImage', {})
            image_type = image.get('PIM_DP_IMAGE_TYPE').lower()
            if image_type in ['wsi']:
                break

        if image is not None:
            self.image_type = image_type
            nbits = image.get('UFS_IMAGE_BLOCK_HEADER_TEMPLATES', [{}])[0].get('UFSImageBlockHeaderTemplate', {}).get('DICOM_BITS_STORED', 16)
            nbits = int(np.ceil(nbits / 8)) * 8
        else:
            self.image_type = ''
            nbits = 16

        self.is_plate = 'screen' in self.image_type or 'plate' in self.image_type or 'wells' in self.image_type

        self.isyntax = ISyntax.open(self.uri)
        self.dimensions = self.isyntax.level_dimensions
        self.widths = [width for width, height in self.isyntax.level_dimensions]
        self.heights = [height for width, height in self.isyntax.level_dimensions]
        self.scales = [1 / downsample for downsample in self.isyntax.level_downsamples]

        # original color channels get converted in pyisyntax package to 8-bit RGBA; convert to RGB
        nbits = 8
        self.nchannels = 3
        self.shapes = [(height, width, self.nchannels) for (width, height) in self.dimensions]
        self.shape = self.shapes[0]
        self.dim_order = 'yxc'
        self.is_rgb_channels = True
        self.dtype = np.dtype(f'uint{nbits}')
        self.pixel_size = {'x': self.isyntax.mpp_x, 'y': self.isyntax.mpp_y}

        self.name = get_filetitle(self.uri)
        return self.metadata

    def is_screen(self):
        return self.is_plate

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_scales(self):
        return self.scales

    def read_array(self, x, y, width, height, level=0):
        rgba = self.isyntax.read_region(x, y, width, height, level)
        alpha = np.atleast_3d(rgba[..., 3] / np.float32(255))
        rgb = (rgba[..., :3] * alpha).astype(np.uint8)
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
        y_pos = np.cumsum([0] + list(y_chunks)[:-1])
        x_pos = np.cumsum([0] + list(x_chunks)[:-1])
        data = da.concatenate(
            [da.concatenate(
                [get_lazy_tile(x, y, width, height, level=level)
                 for x, width in zip(x_pos, x_chunks)], axis=1)
             for y, height in zip(y_pos, y_chunks)], axis=0)
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
        """
        Gets the file title.

        Returns:
            str: Name.
        """
        return self.name

    def get_dim_order(self):
        """
        Returns the dimension order string.

        Returns:
            str: Dimension order.
        """
        return self.dim_order

    def get_pixel_size_um(self):
        """
        Returns the pixel size in micrometers.

        Returns:
            dict: Pixel size dict for x and y.
        """
        return self.pixel_size

    def get_dtype(self):
        """
        Returns the numpy dtype of the image data.

        Returns:
            dtype: Numpy dtype.
        """
        return self.dtype

    def get_position_um(self, well_id=None):
        """
        Returns the position in micrometers (empty for ISyntax).

        Returns:
            dict: Position dict for x and y.
        """
        return {'x': self.isyntax.offset_x, 'y': self.isyntax.offset_y}

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
        return self.is_rgb_channels

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
        total_size = np.prod(self.shape)
        if self.is_plate:
            total_size *= len(self.get_wells()) * len(self.get_fields())
        return total_size

    def close(self):
        self.isyntax.close()
        dask.config.set(scheduler='threads')
