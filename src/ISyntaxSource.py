# Uses https://github.com/anibali/pyisyntax
# which is based on https://github.com/amspath/libisyntax


import dask
import dask.array as da
from isyntax import ISyntax
import numpy as np
from xml.etree import ElementTree

from src.ImageSource import ImageSource
from src.parameters import *
from src.util import get_filetitle, xml_content_to_dict, redimension_data
from src.WindowScanner import WindowScanner


class ISyntaxSource(ImageSource):
    """
    Loads image and metadata from ISyntax format files.
    """
    def init_metadata(self):
        """
        Initializes and loads metadata from the ISyntax file.

        Returns:
            dict: Metadata dictionary.
        """
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

        # original color channels get converted in pyisyntax package to 8-bit RGBA
        nbits = 8
        self.channels = []
        self.nchannels = 4
        self.source_dim_order = 'yxc'
        self.dim_order = 'tczyx'
        self.is_rgb_channels = True

        self.isyntax = ISyntax.open(self.uri)
        self.width, self.height = self.isyntax.dimensions
        self.shape = 1, self.nchannels, 1, self.height, self.width
        self.dtype = np.dtype(f'uint{nbits}')

        return self.metadata

    def is_screen(self):
        """
        Checks if the source is a plate/screen.

        Returns:
            bool: True if plate/screen.
        """
        return self.is_plate

    def get_shape(self):
        """
        Returns the shape of the image data.

        Returns:
            tuple: Shape of the image data.
        """
        return self.shape

    def get_data(self, well_id=None, field_id=None, as_dask=False):
        """
        Gets image data for a specific well and field.

        Returns:
            ndarray: Image data.
        """

        def get_lazy_tile(x, y, width, height):
            lazy_array = dask.delayed(self.isyntax.read_region)(x, y, width, height)
            return da.from_delayed(lazy_array, shape=(height, width, self.nchannels), dtype=self.dtype)

        if as_dask:
            shape = self.shape[-2:]
            y_chunks, x_chunks = da.core.normalize_chunks(TILE_SIZE, shape, dtype=self.dtype)
            rows = []
            x = 0
            y = 0
            for height in y_chunks:
                row = []
                for width in x_chunks:
                    row.append(get_lazy_tile(x, y, width, height))
                    x += width
                rows.append(da.concatenate(row, axis=1))
                y += height
            data = da.concatenate(rows, axis=0)
        else:
            data = self.isyntax.read_region(0, 0, self.width, self.height)

        return redimension_data(data, self.source_dim_order, self.dim_order)

    def get_image_window(self, well_id=None, field_id=None, data=None):
        # For RGB(A) uint8 images don't change color value range
        if not (self.is_rgb_channels and self.dtype == np.uint8):
            level = None
            dims = None
            for level0, dims0 in enumerate(self.isyntax.level_dimensions):
                if np.prod(dims0) < 1e7:
                    level = level0
                    dims = dims0
                    break
            if level is not None:
                window_scanner = WindowScanner()
                data = self.isyntax.read_region(0, 0, dims[0], dims[1], level=level)
                window_scanner.process(data, self.source_dim_order)
                return window_scanner.get_window()
        return [], []

    def get_name(self):
        """
        Gets the file title.

        Returns:
            str: Name.
        """
        return get_filetitle(self.uri)

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
            dict: Pixel size for x and y.
        """
        return {'x': self.isyntax.mpp_x, 'y': self.isyntax.mpp_y}

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
            dict: Empty dict.
        """
        return {}

    def get_channels(self):
        """
        Returns channel metadata.

        Returns:
            list: List of channel dicts.
        """
        return self.channels

    def get_nchannels(self):
        """
        Returns the number of channels.

        Returns:
            int: Number of channels.
        """
        return self.nchannels

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        """
        return self.is_rgb_channels

    def get_rows(self):
        """
        Returns the list of row identifiers (empty for ISyntax).

        Returns:
            list: Empty list.
        """
        return []

    def get_columns(self):
        """
        Returns the list of column identifiers (empty for ISyntax).

        Returns:
            list: Empty list.
        """
        return []

    def get_wells(self):
        """
        Returns the list of well identifiers (empty for ISyntax).

        Returns:
            list: Empty list.
        """
        return []

    def get_time_points(self):
        """
        Returns the list of time points (empty for ISyntax).

        Returns:
            list: Empty list.
        """
        return []

    def get_fields(self):
        """
        Returns the list of field indices (empty for ISyntax).

        Returns:
            list: Empty list.
        """
        return []

    def get_acquisitions(self):
        """
        Returns acquisition metadata (empty for ISyntax).

        Returns:
            list: Empty list.
        """
        return []

    def get_total_data_size(self):
        """
        Returns the estimated total data size.

        Returns:
            int: Total data size in bytes.
        """
        total_size = np.prod(self.shape)
        if self.is_plate:
            total_size *= len(self.get_wells()) * len(self.get_fields())
        return total_size

    def close(self):
        """
        Closes the ISyntax file.
        """
        self.isyntax.close()
