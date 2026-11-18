# https://openslide.org/formats/mirax/

from datetime import datetime
import numpy as np
import openslide

from biomero_converter.TiledSource import TiledSource
from biomero_converter.color_conversion import hexrgb_to_rgba
from biomero_converter.parameters import *
from biomero_converter.util import get_filetitle, get_bits_type


class MiraxSource(TiledSource):
    """
    ImageSource subclass for reading Mirax files using OpenSlide.
    """

    def __init__(self, uri, metadata={}, **kwargs):
        super().__init__(uri, metadata, **kwargs)
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

        name = None
        nbits = 8
        acquisition_datetime = None
        for key, value in self.metadata.items():
            if 'slide_name' in key:
                name = value
            if 'slide_bitdepth' in key:
                nbits = int(value)
            if 'slide_creationdatetime' in key:
                acquisition_datetime = datetime.strptime(value,'%d/%m/%Y %H:%M:%S')
        self.dtype = get_bits_type(nbits)

        # OpenSlide stores microns per pixel in properties
        mpp_x = float(self.metadata.get(openslide.PROPERTY_NAME_MPP_X, 1))
        mpp_y = float(self.metadata.get(openslide.PROPERTY_NAME_MPP_Y, 1))
        self.pixel_size = {'x': mpp_x, 'y': mpp_y}
        background_float = hexrgb_to_rgba(self.metadata.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR, '000000'))[:3]
        self.background = [np.uint8(value * 255) for value in background_float]

        if not name:
            name = get_filetitle(self.uri)
        self.name = name
        self.acquisition_datetime = acquisition_datetime
        self.bits_per_pixel = nbits

        # the scanned region: the bounding box of the (non-empty) tiles
        if openslide.PROPERTY_NAME_BOUNDS_WIDTH in self.metadata:
            self.bounds = [int(self.metadata[key]) for key in (
                openslide.PROPERTY_NAME_BOUNDS_X, openslide.PROPERTY_NAME_BOUNDS_Y,
                openslide.PROPERTY_NAME_BOUNDS_WIDTH, openslide.PROPERTY_NAME_BOUNDS_HEIGHT)]
        self.crop()
        return self.metadata

    def read_rgba(self, x, y, width, height, level=0):
        # OpenSlide (thread-safe) uses (x, y) coordinates in level 0 reference size
        x0 = int(x * self.level_downsamples[level])
        y0 = int(y * self.level_downsamples[level])
        return np.asarray(self.slide.read_region((x0, y0), level, (width, height)))

    def get_position_um(self, well_id=None):
        # the position of the cropped region in the slide (else 0)
        return self.get_crop_position_um()

    def close(self):
        self.slide.close()
        super().close()
