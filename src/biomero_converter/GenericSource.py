import imageio.v3 as iio
import numpy as np
import os.path

from biomero_converter.ImageSource import ImageSource
from biomero_converter.TiffSource import parse_tiff_datetime
from biomero_converter.metadata import get_extra_metadata, field_of_view_pixel_size, get_position_um, \
    is_plausible_pixel_size, stated_pixel_size
from biomero_converter.util import get_filetitle, redimension_data, to_plain_types, without_keys


class GenericSource(ImageSource):
    """
    Loads image and metadata from any (single resolution) image format imageio can read, e.g. JPEG, PNG, GIF.
    """
    def __init__(self, uri, metadata={}):
        """
        Initialize GenericSource.

        Args:
            uri (str): Path to the image file.
            metadata (dict): Optional metadata dictionary.
        """
        super().__init__(uri, metadata)
        try:
            self.image = iio.imopen(uri, 'r')
        except OSError as error:
            raise ValueError(f'Unsupported image format: {uri}') from error

    def init_metadata(self):
        properties = self.image.properties()
        self.metadata = self.image.metadata()
        # vendor metadata in generic form, but what only lays out or encodes the pixel data; as plain data, as
        # Pillow keys some of it by number (Photoshop resources) and holds it as bytes and rationals
        self.acquisition_metadata = to_plain_types(without_keys(get_extra_metadata(self.metadata), GENERIC_LAYOUT_KEYS))

        self.shape = properties.shape
        self.dtype = properties.dtype
        self.dim_order = get_dim_order(self.shape, properties.is_batch)
        self.shapes = [self.shape]
        self.scales = [1]
        self.nchannels = self.shape[self.dim_order.index('c')] if 'c' in self.dim_order else 1
        self.is_rgb_type = self.nchannels in (3, 4) and self.dtype == np.uint8
        self.bits_per_pixel = self.dtype.itemsize * 8

        # only a size the metadata states: the resolution (dpi) of these formats is for printing
        shape = dict(zip(self.dim_order, self.shape))
        pixel_size = {}
        for source in (stated_pixel_size, field_of_view_pixel_size):
            for dim, size in source(self.acquisition_metadata, shape).items():
                pixel_size.setdefault(dim, size)
        self.pixel_size = {dim: size for dim, size in pixel_size.items() if is_plausible_pixel_size(size)}
        self.position = get_position_um(self.acquisition_metadata)

        self.acquisition_datetime = None
        # Exif date the image was taken, else when it was written
        for key in ('DateTimeOriginal', 'DateTime'):
            if isinstance(self.acquisition_metadata.get(key), str):
                try:
                    self.acquisition_datetime = parse_tiff_datetime(self.acquisition_metadata[key])
                    break
                except (ValueError, OverflowError):
                    pass

        self.name = get_filetitle(self.uri)
        return self.metadata

    def is_screen(self):
        return False

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_scales(self):
        return self.scales

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        data = self.image.read()
        return redimension_data(data, self.dim_order, dim_order)

    def get_name(self):
        return self.name

    def get_dim_order(self):
        return self.dim_order

    def get_dtype(self):
        return self.dtype

    def get_pixel_size_um(self):
        if self.pixel_size:
            return self.pixel_size
        else:
            return {'x': 1, 'y': 1}

    def get_position_um(self, well_id=None):
        return self.position

    def get_channels(self):
        if self.is_rgb():
            labels = ['Red', 'Green', 'Blue', 'Alpha']
            colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 1, 1)]
            return [{'label': label, 'color': color} for label, color in zip(labels[:self.nchannels], colors)]
        else:
            return [{'label': f'Channel {index}', 'color': [1, 1, 1, 1]} for index in range(self.nchannels)]

    def get_nchannels(self):
        return self.nchannels

    def is_rgb(self):
        return self.is_rgb_type

    def get_rows(self):
        return []

    def get_columns(self):
        return []

    def get_wells(self):
        return []

    def get_time_points(self):
        if 't' in self.dim_order:
            return list(range(self.shape[self.dim_order.index('t')]))
        return []

    def get_fields(self):
        return []

    def get_acquisitions(self):
        return []

    def get_acquisition_datetime(self):
        # the vendor's acquisition date before the Exif dates
        return self.get_model_acquisition_datetime() or self.acquisition_datetime

    def get_significant_bits(self):
        return self.bits_per_pixel

    def get_acquisition_metadata(self):
        return self.acquisition_metadata

    def close(self):
        self.image.close()


def get_dim_order(shape, is_batch=False):
    """
    Get the dimension order of imageio data: images (frames) first if a batch, then y, x and samples (channels).
    """
    dim_order = 't' if is_batch else ''
    if len(shape) - len(dim_order) > 2:
        dim_order += 'yxc'
    else:
        dim_order += 'yx'
    return dim_order


# metadata (Pillow) that only lays out or encodes the pixel data, or is held expanded beside it (Exif)
GENERIC_LAYOUT_KEYS = ('exif', 'ExifOffset', 'icc_profile', 'mode', 'shape', 'compression', 'progressive',
                       'progression', 'interlace', 'jfif', 'jfif_version', 'jfif_unit', 'jfif_density',
                       'transparency', 'background', 'version')
