import dask.array as da
from datetime import datetime
from enum import Enum
import numpy as np
import os.path
from tifffile import TiffFile, imread, PHOTOMETRIC

from src.ImageSource import ImageSource
from src.ome_tiff_util import metadata_to_dict, read_ome_xml_metadata
from src.parameters import TILE_SIZE
from src.util import convert_to_um, ensure_list, redimension_data, get_filetitle


class TiffSource(ImageSource):
    """
    Loads image and metadata from TIFF or OME-TIFF files.
    """
    def __init__(self, uri, metadata={}):
        """
        Initialize TiffSource.

        Args:
            uri (str): Path to the TIFF file.
            metadata (dict): Optional metadata dictionary.
        """
        super().__init__(uri, metadata)
        image_filename = None
        ext = os.path.splitext(uri)[1].lower()
        if 'tif' in ext:
            image_filename = uri
        elif 'ome' in ext:
            # read metadata
            with open(uri, 'rb') as file:
                self.metadata = metadata_to_dict(file.read().decode())
            # try to open a ome-tiff file
            self.image_filenames = {}
            for image in ensure_list(self.metadata.get('Image', {})):
                filename = image.get('Pixels', {}).get('TiffData', {}).get('UUID', {}).get('FileName')
                if filename:
                    filepath = os.path.join(os.path.dirname(uri), filename)
                    self.image_filenames[image['ID']] = filepath
                    if image_filename is None:
                        image_filename = filepath
        else:
            raise RuntimeError(f'Unsupported tiff extension: {ext}')

        self.tiff = TiffFile(image_filename)

    def init_metadata(self):
        self.is_ome = self.tiff.is_ome
        self.is_imagej = self.tiff.is_imagej
        pixel_size = {}
        position = {}
        channels = []
        microscope_info = {}
        wells = {}
        rows = []
        columns = []
        fields = []
        image_refs = {}

        if self.tiff.series:
            pages = self.tiff.series
            page = pages[0]
        else:
            pages = self.tiff.pages
            page = self.tiff.pages.first
        if hasattr(page, 'levels'):
            pages = page.levels
        self.shapes = [page.shape for page in pages]
        self.shape = page.shape
        self.dim_order = page.axes.lower().replace('s', 'c').replace('r', '')
        x_index, y_index = self.dim_order.index('x'), self.dim_order.index('y')
        self.scales = [float(np.mean([shape[x_index] / self.shape[x_index], shape[y_index] / self.shape[y_index]]))
                       for shape in self.shapes]
        self.is_photometric_rgb = (self.tiff.pages.first.photometric == PHOTOMETRIC.RGB)
        self.nchannels = self.shape[self.dim_order.index('c')] if 'c' in self.dim_order else 1

        if self.is_ome:
            metadata = metadata_to_dict(self.tiff.ome_metadata)
            if metadata and not 'BinaryOnly' in metadata:
                self.metadata = metadata
            (name, is_plate, pixel_size, position, dtype, bits_per_pixel, channels, microscope_info, acquisition_datetime,
             wells, rows, columns, fields, image_refs) = read_ome_xml_metadata(self.metadata)
        else:
            is_plate = False
            if self.is_imagej:
                self.imagej_metadata = self.tiff.imagej_metadata
                pixel_size_unit = self.imagej_metadata.get('unit', '').encode().decode('unicode_escape')
                if 'scales' in self.imagej_metadata:
                    for dim, scale in zip(['x', 'y'], self.imagej_metadata['scales'].split(',')):
                        scale = scale.strip()
                        if scale != '':
                            pixel_size[dim] = convert_to_um(float(scale), pixel_size_unit)
                if 'spacing' in self.imagej_metadata:
                    pixel_size['z'] = convert_to_um(self.imagej_metadata['spacing'], pixel_size_unit)
            self.metadata = tags_to_dict(self.tiff.pages.first.tags)
            name = self.tiff.filename
            if 'DateTime' in self.metadata:
                acquisition_datetime = datetime.strptime(self.metadata['DateTime'],'%Y:%m:%d %H:%M:%S')
            else:
                acquisition_datetime = datetime.fromtimestamp(self.tiff.fstat.st_ctime)
            dtype = page.dtype
            bits_per_pixel = dtype.itemsize * 8
            res_unit = self.metadata.get('ResolutionUnit', '').lower()
            if res_unit == 'none':
                res_unit = ''
            if 'x' not in pixel_size:
                res0 = convert_rational_value(self.metadata.get('XResolution'))
                if res0 is not None and res0 != 0:
                    pixel_size['x'] = convert_to_um(1 / res0, res_unit)
            if 'y' not in pixel_size:
                res0 = convert_rational_value(self.metadata.get('YResolution'))
                if res0 is not None and res0 != 0:
                    pixel_size['y'] = convert_to_um(1 / res0, res_unit)

        if not name:
            name = get_filetitle(self.uri)
        self.name = os.path.splitext(str(name))[0].rstrip('.ome')
        self.acquisition_datetime = acquisition_datetime
        self.is_plate = is_plate
        self.wells = wells
        self.rows = rows
        self.columns = columns
        self.fields = fields
        self.image_refs = image_refs
        self.pixel_size = pixel_size
        self.position = position
        self.channels = channels
        self.dtype = dtype
        self.bits_per_pixel = bits_per_pixel
        self.microscope_info = microscope_info
        return self.metadata

    def is_screen(self):
        return self.is_plate

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_scales(self):
        return self.scales

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        if well_id is not None:
            image_id = self.image_refs[well_id][field_id]
            tiff = TiffFile(self.image_filenames[image_id])
        else:
            tiff = self.tiff
        data = tiff.asarray(level=level)
        return redimension_data(data, self.dim_order, dim_order)

    def get_data_as_dask(self, dim_order, level=0, **kwargs):
        #lazy_array = dask.delayed(imread)(self.uri, level=level)
        #data = da.from_delayed(lazy_array, shape=self.shapes[level], dtype=self.dtype)
        data = da.from_zarr(imread(self.uri, level=level, aszarr=True))
        if data.chunksize == data.shape:
            data = data.rechunk(TILE_SIZE)
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
        return self.channels

    def get_nchannels(self):
        return self.nchannels

    def is_rgb(self):
        return self.is_photometric_rgb

    def get_rows(self):
        return self.rows

    def get_columns(self):
        return self.columns

    def get_wells(self):
        return self.wells

    def get_time_points(self):
        nt = 1
        if 't' in self.dim_order:
            t_index = self.dim_order.index('t')
            nt = self.tiff.pages.first.shape[t_index]
        return list(range(nt))

    def get_fields(self):
        return self.fields

    def get_acquisitions(self):
        return []

    def get_acquisition_datetime(self):
        return self.acquisition_datetime

    def get_significant_bits(self):
        return self.bits_per_pixel

    def get_microscope_info(self):
        return self.microscope_info

    def close(self):
        self.tiff.close()


def tags_to_dict(tags):
    """
    Converts TIFF tags to a dictionary.

    Args:
        tags: TIFF tags object.

    Returns:
        dict: Tag name-value mapping.
    """
    tag_dict = {}
    for tag in tags.values():
        value = tag.value
        if isinstance(value, Enum):
            value = value.name
        tag_dict[tag.name] = value
    return tag_dict


def convert_rational_value(value):
    """
    Converts a rational value tuple to a float.

    Args:
        value (tuple or None): Rational value.

    Returns:
        float or None: Converted value.
    """
    if value is not None and isinstance(value, tuple):
        if value[0] == value[1]:
            value = value[0]
        else:
            value = value[0] / value[1]
    return value
