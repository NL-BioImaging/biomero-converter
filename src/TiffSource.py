import dask.array as da
from datetime import datetime
from enum import Enum
import numpy as np
import os.path
from tifffile import TiffFile, imread, PHOTOMETRIC

from src.ImageSource import ImageSource
from src.color_conversion import int_to_rgba
from src.ome_tiff_util import metadata_to_dict, create_row_col_label
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
            image0 = ensure_list(self.metadata.get('Image', []))[0]
            self.is_plate = 'Plate' in self.metadata
            if self.is_plate:
                plate = self.metadata['Plate']
                name = plate.get('Name')
                rows = set()
                columns = set()
                wells = {}
                fields = []
                image_refs = {}
                for well in ensure_list(plate['Well']):
                    row = create_row_col_label(well['Row'], plate['RowNamingConvention'])
                    column = create_row_col_label(well['Column'], plate['ColumnNamingConvention'])
                    rows.add(row)
                    columns.add(column)
                    label = f'{row}{column}'
                    wells[label] = well['ID']
                    image_refs[label] = {}
                    for sample in ensure_list(well.get('WellSample')):
                        index = sample.get('Index', 0)
                        image_refs[label][str(index)] = sample['ImageRef']['ID']
                        if index not in fields:
                            fields.append(index)
                self.rows = sorted(rows)
                self.columns = list(columns)
                self.wells = list(wells.keys())
                self.fields = fields
                self.image_refs = image_refs
            else:
                name = image0.get('Name')
            if not name:
                name = get_filetitle(self.uri)
            self.acquisition_datetime = image0.get('AcquisitionDate')
            pixels = image0.get('Pixels', {})
            self.dtype = np.dtype(pixels['Type'])
            if 'PhysicalSizeX' in pixels:
                pixel_size['x'] = convert_to_um(float(pixels.get('PhysicalSizeX')), pixels.get('PhysicalSizeXUnit'))
            if 'PhysicalSizeY' in pixels:
                pixel_size['y'] = convert_to_um(float(pixels.get('PhysicalSizeY')), pixels.get('PhysicalSizeYUnit'))
            if 'PhysicalSizeZ' in pixels:
                pixel_size['z'] = convert_to_um(float(pixels.get('PhysicalSizeZ')), pixels.get('PhysicalSizeZUnit'))
            plane = pixels.get('Plane')
            if plane:
                if 'PositionX' in plane:
                    position['x'] = convert_to_um(float(plane.get('PositionX')), plane.get('PositionXUnit'))
                if 'PositionY' in plane:
                    position['y'] = convert_to_um(float(plane.get('PositionY')), plane.get('PositionYUnit'))
                if 'PositionZ' in plane:
                    position['z'] = convert_to_um(float(plane.get('PositionZ')), plane.get('PositionZUnit'))
            for channel0 in ensure_list(pixels.get('Channel')):
                channel = {}
                if 'Name' in channel0:
                    channel['label'] = channel0['Name']
                if 'Color' in channel0:
                    channel['color'] = int_to_rgba(channel0['Color'])
                channels.append(channel)
            if 'SignificantBits' in pixels:
                self.bits_per_pixel = int(pixels['SignificantBits'])
            else:
                self.bits_per_pixel = self.dtype.itemsize * 8
        else:
            self.is_plate = False
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
            name = os.path.splitext(self.tiff.filename)[0]
            if 'DateTime' in self.metadata:
                self.acquisition_datetime = datetime.strptime(self.metadata['DateTime'],'%Y:%m:%d %H:%M:%S')
            else:
                self.acquisition_datetime = datetime.fromtimestamp(self.tiff.fstat.st_ctime)
            self.dtype = page.dtype
            self.bits_per_pixel = self.dtype.itemsize * 8
            res_unit = self.metadata.get('ResolutionUnit', '')
            if isinstance(res_unit, Enum):
                res_unit = res_unit.name
            res_unit = res_unit.lower()
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
        self.name = str(name).rstrip('.tiff').rstrip('.tif').rstrip('.ome')
        self.pixel_size = pixel_size
        self.position = position
        self.channels = channels
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
            index = self.image_refs[well_id][str(field_id)]
            tiff = TiffFile(self.image_filenames[index])
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
        tag_dict[tag.name] = tag.value
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
