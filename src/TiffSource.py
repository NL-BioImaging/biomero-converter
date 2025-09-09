from enum import Enum
import numpy as np
import os.path
from tifffile import TiffFile

from ome_tiff_util import metadata_to_dict
from src.color_conversion import int_to_rgba
from src.ImageSource import ImageSource
from src.util import convert_to_um, ensure_list


class TiffSource(ImageSource):
    def __init__(self, uri, metadata={}):
        super().__init__(uri, metadata)
        image_filename = None
        ext = os.path.splitext(uri)[1].lower()
        if 'tif' in ext:
            image_filename = uri
        elif 'ome' in ext:
            # read metadata
            with open(uri, 'rb') as file:
                self.metadata = metadata_to_dict(file.read().decode())
            # try to open first ome-tiff file
            filename = ensure_list(self.metadata.get('Image', {}))[0].get('Pixels', {}).get('TiffData', {}).get('UUID', {}).get('FileName')
            if filename:
                image_filename = os.path.join(os.path.dirname(uri), filename)
        else:
            raise RuntimeError(f'Unsupported tiff extension: {ext}')

        self.tiff = TiffFile(image_filename)

    def init_metadata(self):
        self.is_ome = self.tiff.is_ome
        self.is_imagej = self.tiff.is_imagej
        pixel_size = {'x': 1, 'y': 1}
        position = {}
        channels = []
        if self.is_ome:
            metadata = metadata_to_dict(self.tiff.ome_metadata)
            if metadata and not 'BinaryOnly' in metadata:
                self.metadata = metadata
            self.is_plate = 'Plate' in self.metadata
            if self.is_plate:
                plate = self.metadata['Plate']
                self.name = plate.get('Name')
                rows = set()
                columns = set()
                wells = {}
                for well in plate['Well']:
                    row = chr(ord('A') + well['Row'])
                    column = well['Column']
                    rows.add(row)
                    columns.add(column)
                    label = f'{row}{column}'
                    wells[label] = well['ID']
                self.rows = sorted(rows)
                self.columns = list(columns)
                self.wells = list(wells.keys())
            else:
                self.name = self.metadata['Image'].get('Name')
            pixels = ensure_list(self.metadata.get('Image', []))[0].get('Pixels', {})
            self.shape = pixels.get('SizeT'), pixels.get('SizeC'), pixels.get('SizeZ'), pixels.get('SizeY'), pixels.get('SizeX')
            self.dim_order = ''.join(reversed(pixels['DimensionOrder'].lower()))
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
                label = channel0.get('Name')
                if label is None:
                    label = ''
                channel = {'label': label}
                color = channel0.get('Color')
                if color is not None:
                    channel['color'] = int_to_rgba(color)
                channels.append(channel)
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
            self.name = os.path.splitext(self.tiff.filename)[0]
            if self.tiff.series:
                page = self.tiff.series[0]
            else:
                page = self.tiff.pages.first
            self.shape = page.shape
            while len(self.shape) < 5:
                self.shape = tuple([1] + list(self.shape))
            self.dim_order = page.axes.lower().replace('s', 'c').replace('r', '')
            self.dtype = page.dtype
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
        self.pixel_size = pixel_size
        self.position = position
        self.channels = channels
        return self.metadata

    def is_screen(self):
        return self.is_plate

    def get_shape(self):
        return self.shape

    def get_data(self, well_id=None, field_id=None):
        data = self.tiff.asarray()
        while data.ndim < len(self.dim_order):
            data = np.expand_dims(data, 0)
        return data

    def get_name(self):
        return self.name

    def get_dim_order(self):
        return self.dim_order

    def get_dtype(self):
        return self.dtype

    def get_pixel_size_um(self):
        return self.pixel_size

    def get_position_um(self, well_id=None):
        return self.position

    def get_channels(self):
        return self.channels

    def get_nchannels(self):
        nchannels = 1
        if 'c' in self.dim_order:
            c_index = self.dim_order.index('c')
            nchannels = self.tiff.pages.first.shape[c_index]
        return nchannels

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
        return nt

    def get_fields(self):
        return self.fields

    def get_acquisitions(self):
        return []

    def get_total_data_size(self):
        total_size = np.prod(self.shape)
        if self.is_plate:
            total_size *= len(self.get_wells()) * len(self.get_fields())
        return total_size

    def close(self):
        self.tiff.close()


def tags_to_dict(tags):
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict


def convert_rational_value(value):
    if value is not None and isinstance(value, tuple):
        if value[0] == value[1]:
            value = value[0]
        else:
            value = value[0] / value[1]
    return value
