from datetime import datetime
import imageio.v3 as iio
import os.path

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
        self.im = iio.imopen(uri, plugin='DICOM', io_mode='rv')
        self.metadata = self.im.metadata()

    def init_metadata(self):
        # default DICOM units are mm
        self.shape = self.metadata.get('shape')
        self.dim_order = 'zyx' if len(self.shape) == 3 else 'yx'
        self.pixel_size = {dim: size for dim, size in zip(self.dim_order, self.metadata.get('sampling'))}
        self.shapes = [self.shape]
        self.scales = [1]
        self.nchannels = 1
        self.position = {dim: size for dim, size in zip(self.dim_order, self.metadata.get('ImagePositionPatient'))}
        date_time = self.metadata.get('AcquisitionDate') + self.metadata.get('AcquisitionTime')
        self.acquisition_datetime = datetime.strptime(date_time, '%Y%m%d%H%M%S')

        self.data = self.im.read()
        self.dtype = self.data.dtype
        self.bits_per_pixel = self.metadata.get('BitsStored', self.dtype.itemsize * 8)

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
        return {dim: size * 1e3 for dim, size in self.pixel_size.items()}

    def get_position_um(self, well_id=None):
        # Not applicable for DICOM
        return {dim: size * 1e3 for dim, size in self.position.items()}

    def get_time_points(self):
        return []

    def get_fields(self):
        return []

    def get_acquisitions(self):
        return []

    def get_acquisition_datetime(self):
        return self.acquisition_datetime

    def get_significant_bits(self):
        return self.bits_per_pixel

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        return redimension_data(self.data, self.dim_order, dim_order)
