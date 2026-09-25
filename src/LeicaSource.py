# https://github.com/cgohlke/liffile
# Image selection & tile scan stitching follow https://github.com/NL-BioImaging/ConvertLeica-Docker

import dask.array as da
import logging
from liffile import LifFile
import numpy as np

from src.ImageSource import ImageSource
from src.parameters import TILE_SIZE
from src.util import get_filetitle, redimension_data, remove_key_prefix


LUT_COLORS = {
    'red': [1, 0, 0, 1],
    'green': [0, 1, 0, 1],
    'blue': [0, 0, 1, 1],
    'cyan': [0, 1, 1, 1],
    'magenta': [1, 0, 1, 1],
    'yellow': [1, 1, 0, 1],
    'gray': [1, 1, 1, 1],
    'grey': [1, 1, 1, 1],
}

IMMERSIONS = {
    'oil': 'Oil',
    'water': 'Water',
    'water dipping': 'WaterDipping',
    'air': 'Air',
    'dry': 'Air',
    'multi': 'Multi',
    'glycerol': 'Glycerol',
    'glyc': 'Glycerol',
}


class LeicaSource(ImageSource):
    """
    Loads image and metadata from Leica LIF, LOF or XLEF files.
    A file can contain multiple images; a single image is selected by UUID or index (default: first image).
    Tile scans (mosaic) are stitched using the tile field positions.
    """
    def __init__(self, uri, metadata={}, image_uuid=None, image_index=None, **kwargs):
        """
        Initialize LeicaSource.

        Args:
            uri (str): Path to the Leica file.
            metadata (dict): Optional metadata dictionary.
            image_uuid (str, optional): UUID of the image to read.
            image_index (int, optional): Index of the image to read (e.g. for older files without UUIDs).
        """
        super().__init__(uri, metadata)
        self.lif = LifFile(uri)
        if image_uuid:
            images = [image for image in self.lif.images if image.uuid == image_uuid]
            if not images:
                raise ValueError(f'Image UUID {image_uuid} not found in {uri}')
            self.image = images[0]
        elif image_index is not None:
            self.image = self.lif.images[int(image_index)]
        else:
            if not self.lif.images:
                raise ValueError(f'No images found in {uri}')
            self.image = self.lif.images[0]

    @staticmethod
    def get_image_count(uri):
        """
        Get the number of images in a Leica file.

        Args:
            uri (str): Path to the Leica file.

        Returns:
            int: Number of images.
        """
        with LifFile(uri) as lif:
            return len(lif.images)

    def init_metadata(self):
        image = self.image
        self.metadata = image.attrs
        sizes = image.sizes
        coords = image.coords

        # map source dimensions to tczyx; RGB samples (S) as channels; other dimensions use index 0
        self.source_dims = image.dims
        self.is_rgb_channels = 'S' in sizes
        unsupported_dims = [dim for dim in self.source_dims if dim not in 'TCZYXSM']
        if unsupported_dims:
            logging.warning(f'Leica image {image.name}: unsupported dimension(s) {unsupported_dims}, using first index')

        self.dim_order = 'tczyx'
        self.tile_size = sizes['Y'], sizes['X']
        self.tile_flip = False, False, False
        self.tile_grid = self._get_tile_grid(sizes.get('M', 1))
        if self.tile_grid is not None:
            ny, nx = len(self.tile_grid), len(self.tile_grid[0])
            height = (ny - 1) * self.tile_step[0] + self.tile_size[0]
            width = (nx - 1) * self.tile_step[1] + self.tile_size[1]
        else:
            height, width = self.tile_size
        nchannels = sizes.get('C', 1) * sizes.get('S', 1)
        self.shape = (sizes.get('T', 1), nchannels, sizes.get('Z', 1), height, width)
        self.shapes = [self.shape]
        self.scales = [1]
        self.nchannels = nchannels
        self.dtype = image.dtype

        self.pixel_size = {}
        self.position = {}
        for dim in 'xyz':
            coord = coords.get(dim.upper())
            if coord is not None and len(coord) > 0:
                self.position[dim] = float(coord[0]) * 1e6
                if len(coord) > 1:
                    self.pixel_size[dim] = abs(float(coord[1] - coord[0])) * 1e6

        channel_elements = sorted(
            image.xml_element.findall('./Data/Image/ImageDescription/Channels/ChannelDescription'),
            key=lambda element: int(element.attrib.get('BytesInc', 0)))
        self.bits_per_pixel = max([int(element.attrib.get('Resolution', 0)) for element in channel_elements],
                                  default=self.dtype.itemsize * 8)
        self.channels = []
        if self.is_rgb_channels:
            for label, color in zip(['Red', 'Green', 'Blue'], [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]):
                self.channels.append({'label': label, 'color': color})
        else:
            labels = coords.get('C', [f'Ch{index}' for index in range(nchannels)])
            # widefield channel names & emission, if matching channels
            widefield_infos = list(image.xml_element.iter('WideFieldChannelInfo'))
            if len(widefield_infos) != len(channel_elements):
                widefield_infos = [None] * len(channel_elements)
            for label, element, widefield_info in zip(labels, channel_elements, widefield_infos):
                channel = {'label': str(label)}
                color = LUT_COLORS.get(element.attrib.get('LUTName', '').lower())
                if color:
                    channel['color'] = color
                if widefield_info is not None:
                    if widefield_info.attrib.get('UserDefName'):
                        channel['label'] = widefield_info.attrib['UserDefName']
                    emission = float(widefield_info.attrib.get('EmissionWavelength', 0))
                    if emission > 0:
                        channel['emission_wavelength'] = emission
                        channel['emission_wavelength_unit'] = 'nm'
                self.channels.append(channel)

        timestamps = image.timestamps
        if timestamps is not None and len(timestamps) > 0:
            self.acquisition_datetime = timestamps[0].astype('datetime64[us]').item()
        else:
            self.acquisition_datetime = None

        self.acquisition_metadata = self._get_acquisition_metadata()
        filetitle = get_filetitle(self.uri)
        image_name = image.path.replace('/', '_')
        self.name = filetitle if image_name == filetitle else f'{filetitle}_{image_name}'
        return self.metadata

    def _get_tile_grid(self, ntiles):
        """
        Create grid of mosaic tile indices from the tile scan field positions.

        Returns:
            list: Nested list [y][x] of mosaic (M) index (or None for empty positions), or None if not a tile scan.
        """
        tilescan = self.image.tilescan
        if ntiles <= 1:
            return None
        if tilescan is None:
            logging.warning(f'Leica image {self.image.name}: mosaic without tile scan info, using first tile')
            return None
        # tiles are flipped / transposed before placing on the grid
        self.tile_flip = tilescan.flip_y, tilescan.flip_x, tilescan.swap_xy
        if tilescan.swap_xy:
            self.tile_size = self.tile_size[::-1]

        # mosaic coordinates index into the tile scan (e.g. for cropped tile scans)
        tile_indices = self.image.coords.get('M', range(ntiles))
        if len(tile_indices) != ntiles or max(tile_indices) >= len(tilescan):
            tile_indices = range(ntiles)
        tiles = [tilescan[int(index)] for index in tile_indices]
        field_xs = [int(tile['field_x']) for tile in tiles]
        field_ys = [int(tile['field_y']) for tile in tiles]
        min_x, min_y = min(field_xs), min(field_ys)
        nx, ny = max(field_xs) - min_x + 1, max(field_ys) - min_y + 1

        # step size (overlap) from stage positions of adjacent tiles
        pixel_size_m = [abs(float(np.diff(self.image.coords[dim][:2])[0])) for dim in 'YX']
        tile_positions = {(field_y, field_x): tile for field_x, field_y, tile in zip(field_xs, field_ys, tiles)}
        step = list(self.tile_size)
        for axis, (offset, pos_key) in enumerate([((1, 0), 'pos_y'), ((0, 1), 'pos_x')]):
            deltas = [abs(float(tile_positions[neighbour][pos_key] - tile[pos_key]))
                      for (field_y, field_x), tile in tile_positions.items()
                      if (neighbour := (field_y + offset[0], field_x + offset[1])) in tile_positions]
            deltas = [delta for delta in deltas if delta > 0]
            if deltas and pixel_size_m[axis] > 0:
                step[axis] = int(round(min(deltas) / pixel_size_m[axis]))
        if step[0] > self.tile_size[0] or step[1] > self.tile_size[1]:
            logging.warning(f'Leica image {self.image.name}: tile scan has negative overlap (gaps), '
                            f'stitching tiles without gaps')
        self.tile_step = [min(max(step[axis], 1), self.tile_size[axis]) for axis in range(2)]

        grid = [[None] * nx for _ in range(ny)]
        for m_index, (field_x, field_y) in enumerate(zip(field_xs, field_ys)):
            grid[field_y - min_y][field_x - min_x] = m_index
        return grid

    def _get_acquisition_metadata(self):
        acquisition_metadata = {'manufacturer': 'Leica Microsystems'}
        hardware_setting = self.metadata.get('HardwareSetting', {})
        if not isinstance(hardware_setting, dict):
            hardware_setting = {}
        # keep hierarchy, without (repeated per sequence) Block settings and without ATL prefix
        hardware_setting = remove_key_prefix({key: value for key, value in hardware_setting.items()
                                              if key != 'Name' and 'Block' not in key}, 'ATL')
        settings = next((value for key, value in hardware_setting.items()
                         if key.endswith('SettingDefinition') and isinstance(value, dict)), {})
        if 'MicroscopeModel' in settings:
            acquisition_metadata['model'] = settings['MicroscopeModel']
        if 'ObjectiveName' in settings:
            acquisition_metadata['objective_name'] = settings['ObjectiveName'].strip()
        if 'Magnification' in settings:
            acquisition_metadata['magnification'] = float(settings['Magnification'])
        if 'NumericalAperture' in settings:
            acquisition_metadata['lens_na'] = float(settings['NumericalAperture'])
        if 'Immersion' in settings:
            acquisition_metadata['immersion'] = IMMERSIONS.get(str(settings['Immersion']).lower(), 'Other')
        if 'RefractionIndex' in settings:
            acquisition_metadata['refractive_index'] = float(settings['RefractionIndex'])
        if hardware_setting:
            acquisition_metadata['HardwareSetting'] = hardware_setting
        return acquisition_metadata

    def _get_source_data(self, as_dask=False):
        """
        Get image data in tczyx order (with mosaic tiles stitched).
        """
        data = self.image.asarray(out='memmap')
        dims = ''.join(self.source_dims)
        # select first index of unsupported dimensions
        data = data[tuple(0 if dim not in 'TCZYXSM' else slice(None) for dim in dims)]
        dims = ''.join(dim for dim in dims if dim in 'TCZYXSM')
        if as_dask:
            data = da.from_array(data, chunks=tuple(TILE_SIZE if dim in 'YX' else 1 for dim in dims))
        if 'S' in dims:
            # RGB samples as channels
            if 'C' in dims:
                # move samples next to channels and merge
                new_dims = dims.replace('S', '').replace('C', 'CS')
                data = redimension_data(data, dims.lower(), new_dims.lower())
                c_index = new_dims.index('C')
                data = data.reshape(data.shape[:c_index] + (-1,) + data.shape[c_index + 2:])
                dims = new_dims.replace('S', '')
            else:
                dims = dims.replace('S', 'C')
        order = 'm' + self.dim_order if 'M' in dims else self.dim_order
        data = redimension_data(data, dims.lower(), order)
        if 'M' in dims:
            data = self._stitch_tiles(data, as_dask=as_dask)
        return data

    def _stitch_tiles(self, data, as_dask=False):
        """
        Stitch mosaic tiles (first axis) along the last two (yx) axes.
        """
        if self.tile_grid is None:
            return data[0]
        lib = da if as_dask else np
        flip_y, flip_x, swap_xy = self.tile_flip
        ny, nx = len(self.tile_grid), len(self.tile_grid[0])
        rows = []
        for y, grid_row in enumerate(self.tile_grid):
            row = []
            for x, m_index in enumerate(grid_row):
                # crop overlap, except for last row/column
                height = self.tile_step[0] if y < ny - 1 else self.tile_size[0]
                width = self.tile_step[1] if x < nx - 1 else self.tile_size[1]
                if m_index is not None:
                    tile = data[m_index]
                    if flip_y:
                        tile = tile[..., ::-1, :]
                    if flip_x:
                        tile = tile[..., ::-1]
                    if swap_xy:
                        tile = lib.swapaxes(tile, -1, -2)
                    tile = tile[..., :height, :width]
                else:
                    tile = lib.zeros(data.shape[1:-2] + (height, width), dtype=data.dtype)
                row.append(tile)
            rows.append(row)
        return lib.block(rows)

    def is_screen(self):
        return False

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_scales(self):
        return self.scales

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        data = np.asarray(self._get_source_data())
        return redimension_data(data, self.dim_order, dim_order)

    def get_data_as_dask(self, dim_order, level=0, **kwargs):
        data = self._get_source_data(as_dask=True)
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
        return self.is_rgb_channels

    def get_rows(self):
        return []

    def get_columns(self):
        return []

    def get_wells(self):
        return []

    def get_time_points(self):
        return list(range(self.shape[0]))

    def get_fields(self):
        return []

    def get_acquisitions(self):
        return []

    def get_acquisition_datetime(self):
        return self.acquisition_datetime

    def get_significant_bits(self):
        return self.bits_per_pixel

    def get_acquisition_metadata(self):
        return self.acquisition_metadata

    def close(self):
        self.lif.close()
