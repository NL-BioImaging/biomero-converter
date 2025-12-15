from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os.path
import skimage.transform as sk_transform

from src.ImageSource import ImageSource
from src.ome_zarr_util import *
from src.parameters import TILE_SIZE
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
        else:
            self.name = self.metadata.get('name', '')
            self.data = image_node.data
            self.heights = [data.shape[y_index] for data in self.data]
            self.widths = [data.shape[x_index] for data in self.data]
        if not self.name:
            self.name = get_filetitle(self.uri)
        self.name = str(self.name).rstrip('.ome')
        self.shape = image_node.data[0].shape
        self.dtype = image_node.data[0].dtype

    def is_screen(self):
        return self.is_plate

    def get_shape(self):
        return self.shape

    def get_scales(self):
        return self.scales

    def get_data(self, well_id=None, field_id=None, level=0, **kwargs):
        if well_id is None and field_id is None:
            return self.data[level]
        else:
            _, nodes = self._get_reader(self.paths[well_id][field_id])
            return nodes[0].data[level]

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
        metadata = self._get_metadata(self.paths[well_id][0])
        for transforms in metadata['coordinateTransformations'][0]:
            if transforms['type'] == 'translation':
                return {dim:value for dim, value in zip(self.dim_order, transforms['translation'])}
        return {}

    def get_channels(self):
        channels = []
        colormaps = self.metadata['colormap']
        for channeli, channel_name in enumerate(self.metadata['channel_names']):
            channel = {'label': channel_name}
            if channeli < len(colormaps):
                channel['color'] = colormaps[channeli][-1]
            channels.append(channel)
        return channels

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

    def get_total_data_size(self):
        total_size = np.prod(self.shape)
        if self.is_plate:
            total_size *= len(self.get_wells()) * len(self.get_fields())
        return total_size
