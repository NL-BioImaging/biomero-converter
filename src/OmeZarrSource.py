from datetime import datetime
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os.path
import skimage.transform as sk_transform

from src.ImageSource import ImageSource
from src.ome_tiff_util import metadata_to_dict, read_ome_xml_metadata
from src.ome_zarr_util import *
from src.parameters import *
from src.util import convert_to_um, get_level_from_scale, redimension_data, get_filetitle, get_numpy_data


class OmeZarrSource(ImageSource):

    def _get_reader(self, add_path=None):
        uri = self.uri
        if add_path:
            uri = os.path.join(uri, add_path)
        location = parse_url(uri)
        if location is None:
            raise FileNotFoundError(f'Error parsing ome-zarr file {uri}')
        reader = Reader(location)
        nodes = list(reader())
        return reader, nodes

    def _get_metadata(self, add_path=None):
        metadata = {}
        _, nodes = self._get_reader(add_path)
        if len(nodes) > 0:
            metadata = nodes[0].metadata
        return metadata

    def init_metadata(self):
        reader, nodes = self._get_reader()
        if 'bioformats2raw.layout' in reader.zarr.root_attrs:
            # TODO: use paths provided in metadata
            reader, nodes = self._get_reader('0')
        # nodes may include images, labels etc
        if len(nodes) == 0:
            raise FileNotFoundError(f'No image data found in ome-zarr file {self.uri}')
        # first node will be the image pixel data
        image_node = nodes[0]
        self.metadata = image_node.metadata
        # channel metadata from ome-zarr-py limited; get from root_attrs manually
        #self.root_metadata = reader.zarr.root_attrs

        axes = self.metadata.get('axes', [])
        self.dim_order = ''.join([axis.get('name') for axis in axes])
        units = {axis['name']: axis['unit'] for axis in axes if 'unit' in axis}
        self.plate = self.metadata.get('metadata', {}).get('plate')
        self.is_plate = self.plate is not None

        scales = [transform['scale'] for transform_set in self.metadata['coordinateTransformations']
                  for transform in transform_set if transform['type'] == 'scale']
        self.pixel_size = {dim: convert_to_um(pixel_size, units.get(dim, '')) for dim, pixel_size
                           in zip(self.dim_order, scales[0]) if dim in 'xyz'}
        x_index, y_index = self.dim_order.index('x'), self.dim_order.index('y')
        scale0 = np.mean([scales[0][x_index] + scales[0][y_index]])
        self.scales = [float(scale0 / np.mean([scale[x_index] + scale[y_index]])) for scale in scales]
        if self.is_plate:
            self.name = self.plate.get('name', '')
            self.rows = [row['name'] for row in self.plate.get('rows', [])]
            self.columns = [column['name'] for column in self.plate.get('columns', [])]
            self.wells = {well['path'].replace('/', ''): well['path'] for well in self.plate.get('wells')}
            self.fields = list(range(self.plate.get('field_count', 0)))
            self.paths = {well_id: {field: f'{well_path}/{field}' for field in self.fields} for well_id, well_path in self.wells.items()}
            self.acquisitions = self.plate.get('acquisitions', [])
            self.acquisition_datetime = datetime.fromisoformat(self.acquisitions[0]['date_created'])
            self.data = None    # data will be read per plate well
        else:
            self.name = self.metadata.get('name', '')
            self.acquisition_datetime = datetime.fromtimestamp(os.path.getctime(self.uri))
            self.data = image_node.data
        if not self.name:
            self.name = get_filetitle(self.uri)
        self.name = str(self.name).rstrip('.ome')

        self.shapes = [data.shape for data in image_node.data]
        self.shape = self.shapes[0]
        self.heights = [shape[y_index] for shape in self.shapes]
        self.widths = [shape[x_index] for shape in self.shapes]
        self.dtype = image_node.data[0].dtype
        self.bits_per_pixel = self.dtype.itemsize * 8

        self.channels = []
        colormaps = self.metadata.get('colormap', [])
        for channeli, channel_name in enumerate(self.metadata.get('channel_names', [])):
            channel = {'label': channel_name}
            if channeli < len(colormaps):
                channel['color'] = colormaps[channeli][-1]
            self.channels.append(channel)

        ome_xml_path = image_node.zarr.subpath(os.path.join(OME_DIR, OME_FILE))
        if os.path.exists(ome_xml_path):
            ome_xml_metadata = open(ome_xml_path, encoding='utf-8').read()
            ome_metadata = metadata_to_dict(ome_xml_metadata)
            (name, is_plate, pixel_size, position, dtype, bits_per_pixel, channels, microscope_info,
             acquisition_datetime,
             wells, rows, columns, fields, image_refs) = read_ome_xml_metadata(ome_metadata)
            for channel, ome_channel in zip(self.channels, channels):
                for key, value in ome_channel.items():
                    if key not in channel:
                        channel[key] = value
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
        if well_id is None and field_id is None:
            data = self.data[level]
        else:
            _, nodes = self._get_reader(self.paths[well_id][field_id])
            data = nodes[0].data[level]
        return redimension_data(data, self.dim_order, dim_order)

    def get_data_as_generator(self, dim_order, **kwargs):
        def data_generator(scale=1):
            level, rescale = get_level_from_scale(self.scales, scale)
            level_data = self.data[level]
            read_size = int(TILE_SIZE / rescale)
            nz = self.shape[self.dim_order.index('z')] if 'z' in self.dim_order else 1
            for t in range(len(self.get_time_points())):
                for c in range(self.get_nchannels()):
                    for z in range(nz):
                        for y in range(0, self.heights[level], read_size):
                            for x in range(0, self.widths[level], read_size):
                                data = get_numpy_data(level_data, dim_order, t, c, z, y, x, read_size, read_size)
                                if rescale != 1:
                                    data = sk_transform.resize(data,
                                                               (np.array(data.shape) * rescale).astype(int),
                                                               preserve_range=True).astype(data.dtype)
                                yield redimension_data(data, self.dim_order, dim_order)
        return data_generator

    def get_image_window(self, window_scanner, well_id=None, field_id=None, data=None):
        if well_id is None and field_id is None:
            metadata = self.metadata
        else:
            metadata = self._get_metadata(self.paths[well_id][field_id])
        window = np.transpose(metadata.get('contrast_limits', ([], [])))
        return window

    def get_name(self):
        return self.name

    def get_dim_order(self):
        return self.dim_order

    def get_dtype(self):
        return self.dtype

    def get_pixel_size_um(self):
        return self.pixel_size

    def get_position_um(self, well_id=None):
        if well_id is not None:
            metadata = self._get_metadata(self.paths[well_id][0])
        else:
            metadata = self.metadata
        for transforms in metadata['coordinateTransformations'][0]:
            if transforms['type'] == 'translation':
                return {dim:value for dim, value in zip(self.dim_order, transforms['translation'])}
        return {}

    def get_channels(self):
        return self.channels

    def get_nchannels(self):
        return self.shape[self.dim_order.index('c')] if 'c' in self.dim_order else 1

    def is_rgb(self):
        return self.get_nchannels() in (3, 4)

    def get_rows(self):
        return self.rows

    def get_columns(self):
        return self.columns

    def get_wells(self):
        return self.wells

    def get_time_points(self):
        nt = self.shape[self.dim_order.index('t')] if 't' in self.dim_order else 1
        return list(range(nt))

    def get_fields(self):
        return self.fields

    def get_acquisitions(self):
        return self.acquisitions

    def get_acquisition_datetime(self):
        return self.acquisition_datetime

    def get_significant_bits(self):
        return self.bits_per_pixel

    def get_microscope_info(self):
        return self.microscope_info
