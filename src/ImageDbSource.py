# based on https://github.com/Cellular-Imaging-Amsterdam-UMC/crxReader-Python
# which is based on https://github.com/Cellular-Imaging-Amsterdam-UMC/crxReader
# Screen Plate Well (SPW) - High Content Screening (HCS) https://ome-model.readthedocs.io/en/stable/developers/screen-plate-well.html

import numpy as np

from src.color_conversion import hexrgb_to_rgba
from src.DbReader import DbReader
from src.ImageSource import ImageSource
from src.util import *


class ImageDbSource(ImageSource):
    """
    Loads image and metadata from a database source for high-content screening.
    """
    def __init__(self, uri, metadata={}):
        super().__init__(uri, metadata)
        self.db = DbReader(self.uri)
        self.data = None
        self.data_well_id = None
        self.data_level = None
        self.dim_order = 'tczyx'

    def init_metadata(self):
        self._get_time_series_info()
        self._get_experiment_metadata()
        self._get_well_info()
        self._get_image_info()
        self._get_sizes()
        return self.metadata

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_scales(self):
        return self.scales

    def _get_time_series_info(self):
        """
        Loads time series and image file info into metadata.
        """
        time_series_ids = sorted(self.db.fetch_all('SELECT DISTINCT TimeSeriesElementId FROM SourceImageBase', return_dicts=False))
        self.time_points = time_series_ids

        level_ids = sorted(self.db.fetch_all('SELECT DISTINCT level FROM SourceImageBase', return_dicts=False))
        self.levels = level_ids

        image_files = {time_series_id: os.path.join(os.path.dirname(self.uri), f'images-{time_series_id}.db')
                       for time_series_id in time_series_ids}
        self.image_files = image_files

    def _get_experiment_metadata(self):
        """
        Loads experiment metadata and acquisition info into metadata.
        """
        creation_info = self.db.fetch_all('SELECT DateCreated, Creator, Name FROM ExperimentBase')[0]
        creation_info['DateCreated'] = convert_dotnet_ticks_to_datetime(creation_info['DateCreated'])
        self.metadata.update(creation_info)

        acquisitions = self.db.fetch_all('SELECT Name, Description, DateCreated, DateModified FROM AcquisitionExp')
        for acquisition in acquisitions:
            acquisition['DateCreated'] = convert_dotnet_ticks_to_datetime(acquisition['DateCreated'])
            acquisition['DateModified'] = convert_dotnet_ticks_to_datetime(acquisition['DateModified'])
        self.acquisitions = acquisitions

    def _get_well_info(self):
        """
        Loads well and channel information into metadata.
        """
        well_info = self.db.fetch_all('''
            SELECT SensorSizeYPixels, SensorSizeXPixels, Objective, PixelSizeUm, SensorBitness, SitesX, SitesY
            FROM AcquisitionExp, AutomaticZonesParametersExp
        ''')[0]

        # Filter multiple duplicate channel entries
        channel_infos = self.db.fetch_all('''
            SELECT DISTINCT ChannelNumber, Emission, Excitation, Dye, Color
            FROM ImagechannelExp
            ORDER BY ChannelNumber
        ''')
        self.channels = channel_infos
        self.nchannels = len(channel_infos)

        wells = self.db.fetch_all('SELECT DISTINCT Name FROM Well')
        zone_names = [well['Name'] for well in wells]
        rows = set()
        cols = set()
        for zone_name in zone_names:
            row, col = split_well_name(zone_name)
            rows.add(row)
            cols.add(col)
        self.rows = sorted(list(rows))
        self.columns = sorted(list(cols), key=lambda x: int(x))
        nfields = well_info['SitesX'] * well_info['SitesY'] * well_info.get('SitesZ', 1)
        self.fields = list(range(nfields))
        self.well_info = well_info
        self.metadata['well_info'] = well_info

        image_wells = self.db.fetch_all('SELECT Name, ZoneIndex, CoordX, CoordY FROM Well WHERE HasImages = 1')
        self.wells = dict(sorted({well['Name']: well for well in image_wells}.items(),
                                             key=lambda x: split_well_name(x[0], col_as_int=True)))
        self.metadata['wells'] = self.wells
        self.pixel_size = well_info.get('PixelSizeUm', 1)

    def _get_image_info(self):
        """
        Loads image bit depth and dtype info into metadata.
        """
        bits_per_pixel = self.db.fetch_all('SELECT DISTINCT BitsPerPixel FROM SourceImageBase', return_dicts=False)[0]
        self.bits_per_pixel = bits_per_pixel
        bits_per_pixel = int(np.ceil(bits_per_pixel / 8)) * 8
        if bits_per_pixel == 24:
            bits_per_pixel = 32
        self.dtype = np.dtype(f'uint{bits_per_pixel}')

    def _get_sizes(self):
        """
        Calculates and stores image shape and estimated data size.
        """
        shapes = []
        scales = []
        widths = []
        heights = []
        width0, height0 = self.well_info['SensorSizeXPixels'], self.well_info['SensorSizeYPixels']
        sizex0, sizey0 = None, None
        # Iterate through levels to get level size factor (SourceImageBase contains field-composite images)
        for level in self.levels:
            level_info = self.db.fetch_all(
                'SELECT MAX(CoordX + SizeX) as width, MAX(CoordY + SizeY) as height FROM SourceImageBase WHERE level = ?',
                [level])
            sizex, sizey = level_info[0]['width'], level_info[0]['height']
            if level == 0:
                sizex0, sizey0 = sizex, sizey
            width, height = width0 * sizex // sizex0, height0 * sizey // sizey0
            widths.append(width)
            heights.append(height)
            shape = len(self.time_points), self.nchannels, 1, height, width
            scale = np.mean([width / widths[0], height / heights[0]])
            shapes.append(shape)
            scales.append(scale)
        self.widths = widths
        self.heights = heights
        self.shape = shapes[0]
        self.shapes = shapes
        self.scales = scales
        self.max_data_size = np.prod(self.shape) * self.dtype.itemsize * len(self.wells) * len(self.fields)

    def _read_well_info(self, well_id, channel=None, time_point=None, level=0):
        """
        Reads image info for a specific well, optionally filtered by channel and time point.

        Args:
            well_id (str): Well identifier.
            channel (int, optional): Channel ID.
            time_point (int, optional): Time point ID.
            level (int, optional): Image level index.

        Returns:
            list: Well image info dictionaries.
        """
        well_id = strip_leading_zeros(well_id)
        well_ids = self.wells

        if well_id not in well_ids:
            raise ValueError(f'Invalid Well: {well_id}. Available values: {well_ids}')

        zone_index = well_ids[well_id]['ZoneIndex']
        well_info = self.db.fetch_all('''
            SELECT *
            FROM SourceImageBase
            WHERE ZoneIndex = ? AND level = ?
            ORDER BY CoordX ASC, CoordY ASC
        ''', [zone_index, level])

        if channel is not None:
             well_info = [info for info in well_info if info['ChannelId'] == channel]
        if time_point is not None:
             well_info = [info for info in well_info if info['TimeSeriesElementId'] == time_point]
        if not well_info:
            raise ValueError(f'No data found for well {well_id}')
        return well_info

    def _assemble_image_data(self, well_info):
        """
        Assembles image data array using well info.

        Args:
            well_info (list): List of well image info dicts.
        """
        well_info = np.asarray(well_info)
        xmax = np.max([info['CoordX'] + info['SizeX'] for info in well_info])
        ymax = np.max([info['CoordY'] + info['SizeY'] for info in well_info])
        zmax = np.max([info.get('CoordZ', 0) + info.get('SizeZ', 1) for info in well_info])
        nc = len(set([info['ChannelId'] for info in well_info]))
        nt = len(set([info['TimeSeriesElementId'] for info in well_info]))
        data = np.zeros((nt, nc, zmax, ymax, xmax), dtype=self.dtype)

        for timei, time_id in enumerate(self.time_points):
            image_file = self.image_files[time_id]
            with open(image_file, 'rb') as fid:
                for info in well_info:
                    if info['TimeSeriesElementId'] == time_id:
                        fid.seek(info['ImageIndex'])
                        coordx, coordy, coordz = info['CoordX'], info['CoordY'], info.get('CoordZ', 0)
                        sizex, sizey, sizez = info['SizeX'], info['SizeY'], info.get('SizeZ', 1)
                        channeli = info['ChannelId']
                        tile = np.fromfile(fid, dtype=self.dtype, count=sizez * sizey * sizex)
                        data[timei, channeli, coordz:coordz + sizez, coordy:coordy + sizey, coordx:coordx + sizex] = tile.reshape((sizez, sizey, sizex))

        self.data = data

    def _extract_site(self, site_id=None):
        """
        Extracts image data for a specific site or all sites.

        Args:
            site_id (int, optional): Site index. If None, returns all data.

        Returns:
            ndarray or list: Image data for the site(s).
        """
        well_info = self.well_info
        sitesx = well_info['SitesX']
        sitesy = well_info['SitesY']
        sitesz = well_info.get('SitesZ', 1)
        nfields = len(self.fields)
        sizex = well_info['SensorSizeXPixels']
        sizey = well_info['SensorSizeYPixels']
        sizez = well_info.get('SensorSizeZPixels', 1)

        if site_id is None:
            # Return full image data
            return self.data

        site_id = int(site_id)
        if site_id < 0:
            # Return list of all fields
            data = []
            for zi in range(sitesz):
                for yi in range(sitesy):
                    for xi in range(sitesx):
                        startx = xi * sizex
                        starty = yi * sizey
                        startz = zi * sizez
                        data.append(self.data[..., startz:startz + sizez, starty:starty + sizey, startx:startx + sizex])
            return data
        elif 0 <= site_id < nfields:
            # Return specific site
            xi = site_id % sitesx
            yi = (site_id // sitesx) % sitesy
            zi = site_id // sitesx // sitesy
            startx = xi * sizex
            starty = yi * sizey
            startz = zi * sizez
            return self.data[..., startz:startz + sizez, starty:starty + sizey, startx:startx + sizex]
        else:
            raise ValueError(f'Invalid site: {site_id}')

    def is_screen(self):
        return len(self.wells) > 0

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        if well_id != self.data_well_id and level != self.data_level:
            self._assemble_image_data(self._read_well_info(well_id, level=level))
            self.data_well_id = well_id
            self.data_level = level
        return redimension_data(self._extract_site(field_id), self.dim_order, dim_order)

    def get_name(self):
        name = self.metadata.get('Name')
        if not name:
            name = splitall(os.path.splitext(self.uri)[0])[-2]
        return name

    def get_rows(self):
        return self.rows

    def get_columns(self):
        return self.columns

    def get_wells(self):
        return list(self.wells)

    def get_time_points(self):
        return self.time_points

    def get_fields(self):
        return self.fields

    def get_dim_order(self):
        return self.dim_order

    def get_dtype(self):
        return self.dtype

    def get_pixel_size_um(self):
        return {'x': self.pixel_size, 'y': self.pixel_size}

    def get_position_um(self, well_id=None, level=0):
        well = self.wells[well_id]
        x = well.get('CoordX', 0) * self.widths[level] * self.pixel_size
        y = well.get('CoordY', 0) * self.heights[level] * self.pixel_size
        return {'x': x, 'y': y}

    def get_channels(self):
        channels = []
        for channel0 in self.channels:
            channel = {}
            if 'Dye' in channel0 and channel0['Dye']:
                channel['label'] = channel0['Dye']
            if 'Color' in channel0:
                channel['color'] = hexrgb_to_rgba(channel0['Color'].lstrip('#'))
            channels.append(channel)
        return channels

    def get_nchannels(self):
        return max(self.nchannels, 1)

    def is_rgb(self):
        return False

    def get_acquisitions(self):
        acquisitions = []
        for index, acq in enumerate(self.acquisitions):
            acquisitions.append({
                'id': index,
                'name': acq['Name'],
                'description': acq['Description'],
                'date_created': acq['DateCreated'].isoformat(),
                'date_modified': acq['DateModified'].isoformat()
            })
        return acquisitions

    def get_total_data_size(self):
        return self.max_data_size

    def print_timepoint_well_matrix(self):
        s = ''

        time_points = self.time_points
        wells = [well for well in self.wells]

        well_matrix = []
        for timepoint in time_points:
            wells_at_timepoint = self.db.fetch_all('''
                SELECT DISTINCT Well.Name FROM SourceImageBase
                JOIN Well ON SourceImageBase.ZoneIndex = Well.ZoneIndex
                WHERE TimeSeriesElementId = ?
            ''', [timepoint], return_dicts=False)

            row = ['+' if well in wells_at_timepoint else ' ' for well in wells]
            well_matrix.append(row)

        header = ' '.join([pad_leading_zero(well) for well in wells])
        s += 'Timepoint ' + header + '\n'
        for idx, row in enumerate(well_matrix):
            s += f'{time_points[idx]:9}  ' + '   '.join(row) + '\n'
        return s

    def close(self):
        """
        Closes the database connection.
        """
        self.db.close()
