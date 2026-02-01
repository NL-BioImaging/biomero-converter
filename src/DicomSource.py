import os.path

import imageio.v3 as iio

from src.ImageSource import ImageSource
from src.util import get_filetitle, redimension_data


class DicomSource(ImageSource):
    """
    ImageSource subclass for reading DICOM files using pydicom.
    """

    def __init__(self, uri, metadata={}):
        super().__init__(uri, metadata)
        uri2 = os.path.splitext(uri)[0]
        if not os.path.exists(uri) and os.path.exists(uri2):
            uri = uri2
            self.uri = uri
        io_mode = 'rv' if os.path.isdir(uri) else 'r'
        self.im = iio.imopen(uri, plugin='DICOM', io_mode=io_mode)
        self.metadata = self.im.metadata()

    def init_metadata(self):
        self.shape = self.metadata.get('shape')
        self.dim_order = 'zyx' if len(self.shape) == 3 else 'yx'
        self.pixel_size = {dim: size for dim, size in zip(self.dim_order, self.metadata.get('sampling'))}
        self.shapes = [self.shape]
        self.scales = [1]
        self.nchannels = 1
        self.dtype = iio.imopen(self.uri, plugin='DICOM', io_mode='r').read().dtype

        name = self.metadata.get('SeriesInstanceUID').split('.')[-1]
        if not name:
            name = get_filetitle(self.uri)
        self.name = name

        return self.metadata

    def is_screen(self):
        # DICOM files are not multi-well screens
        return False

    def is_rgb(self):
        return self.get_nchannels() in (3, 4)

    def get_name(self):
        return self.name

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_dtype(self):
        return self.dtype

    def get_scales(self):
        return self.scales

    def get_dim_order(self):
        return self.dim_order

    def get_channels(self):
        return [{'label': f'Channel {index}', 'color': [1, 1, 1, 1]} for index in range(self.nchannels)]

    def get_nchannels(self):
        return self.nchannels

    def get_pixel_size_um(self):
        return self.pixel_size

    def get_position_um(self, well_id=None):
        # Not applicable for DICOM
        return {'x': 0, 'y': 0}

    def get_time_points(self):
        return []

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        data = self.im.read()
        return redimension_data(data, self.dim_order, dim_order)
