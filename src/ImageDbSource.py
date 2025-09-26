# based on https://github.com/Cellular-Imaging-Amsterdam-UMC/crxReader-Python
# which is based on https://github.com/Cellular-Imaging-Amsterdam-UMC/crxReader
# Screen Plate Well (SPW) - High Content Screening (HCS) https://ome-model.readthedocs.io/en/stable/developers/screen-plate-well.html

import numpy as np

from src.color_conversion import hexrgb_to_rgba
from src.DbReader import DBReader
from src.ImageSource import ImageSource
from src.util import *
from src.WindowScanner import WindowScanner


class ImageDbSource(ImageSource):
    """
    Loads image and metadata from a database source for high-content screening.
    """
    def __init__(self, uri, metadata={}):
        """
        Initialize ImageDbSource.

        Args:
            uri (str): Path to the database file.
            metadata (dict): Optional metadata dictionary.
        """
        super().__init__(uri, metadata)
        self.db = DBReader(self.uri)
        self.data = None
        self.data_well_id = None
        self.dim_order = 'tczyx'

    def init_metadata(self):
        """
        Initializes and loads metadata from the database.

        Returns:
            dict: Metadata dictionary.
        """
        self._get_time_series_info()
        self._get_experiment_metadata()
        self._get_well_info()
        self._get_image_info()
        self._get_sizes()
        return self.metadata

    def get_shape(self):
        """
        Returns the shape of the image data.

        Returns:
            tuple: Shape of the image data.
        """
        return self.shape

    def _get_time_series_info(self):
        """
        Loads time series and image file info into metadata.
        """
        time_series_ids = sorted(self.db.fetch_all('SELECT DISTINCT TimeSeriesElementId FROM SourceImageBase', return_dicts=False))
        self.metadata['time_points'] = time_series_ids

        level_ids = sorted(self.db.fetch_all('SELECT DISTINCT level FROM SourceImageBase', return_dicts=False))
        self.metadata['levels'] = level_ids

        image_files = {time_series_id: os.path.join(os.path.dirname(self.uri), f'images-{time_series_id}.db')
                       for time_series_id in time_series_ids}
        self.metadata['image_files'] = image_files

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
        self.metadata['acquisitions'] = acquisitions

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
        self.metadata['channels'] = channel_infos
        self.metadata['num_channels'] = len(channel_infos)

        wells = self.db.fetch_all('SELECT DISTINCT Name FROM Well')
        zone_names = [well['Name'] for well in wells]
        rows = set()
        cols = set()
        for zone_name in zone_names:
            row, col = split_well_name(zone_name)
            rows.add(row)
            cols.add(col)
        well_info['rows'] = sorted(list(rows))
        well_info['columns'] = sorted(list(cols), key=lambda x: int(x))
        num_sites = well_info['SitesX'] * well_info['SitesY']
        well_info['num_sites'] = num_sites
        well_info['fields'] = list(range(num_sites))

        image_wells = self.db.fetch_all('SELECT Name, ZoneIndex, CoordX, CoordY FROM Well WHERE HasImages = 1')
        self.metadata['wells'] = dict(sorted({well['Name']: well for well in image_wells}.items(),
                                             key=lambda x: split_well_name(x[0], col_as_int=True)))

        xmax, ymax = 0, 0
        for well_id in self.metadata['wells']:
            well_data = self._read_well_info(well_id)
            xmax = max(xmax, np.max([info['CoordX'] + info['SizeX'] for info in well_data]))
            ymax = max(ymax, np.max([info['CoordY'] + info['SizeY'] for info in well_data]))
        pixel_size = well_info.get('PixelSizeUm', 1)
        well_info['max_sizex_um'] = xmax * pixel_size
        well_info['max_sizey_um'] = ymax * pixel_size

        self.metadata['well_info'] = well_info

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
        well_info = self.metadata['well_info']
        nbytes = self.dtype.itemsize
        self.shape = len(self.metadata['time_points']), self.metadata['num_channels'], 1, well_info['SensorSizeYPixels'], well_info['SensorSizeXPixels']
        max_data_size = np.prod(self.shape) * nbytes * len(self.metadata['wells']) * well_info['num_sites']
        self.metadata['max_data_size'] = max_data_size

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
        well_ids = self.metadata.get('wells', {})

        if well_id not in well_ids:
            raise ValueError(f'Invalid Well: {well_id}. Available values: {well_ids}')

        zone_index = well_ids[well_id]['ZoneIndex']
        well_info = self.db.fetch_all('''
            SELECT *
            FROM SourceImageBase
            WHERE ZoneIndex = ? AND level = ?
            ORDER BY CoordX ASC, CoordY ASC
        ''', (zone_index, level))

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
        nt = len(self.metadata['time_points'])
        data = np.zeros((nt, nc, zmax, ymax, xmax), dtype=self.dtype)

        for timei, time_id in enumerate(self.metadata['time_points']):
            image_file = self.metadata['image_files'][time_id]
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
        well_info = self.metadata['well_info']
        sitesx = well_info['SitesX']
        sitesy = well_info['SitesY']
        sitesz = well_info.get('SitesZ', 1)
        num_sites = well_info['num_sites']
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
        elif 0 <= site_id < num_sites:
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
        """
        Checks if the source is a screen (has wells).

        Returns:
            bool: True if wells exist.
        """
        return len(self.metadata['wells']) > 0

    def get_data(self, well_id=None, field_id=None, as_dask=False):
        """
        Gets image data for a specific well and field.

        Returns:
            ndarray: Image data.
        """
        if well_id != self.data_well_id:
            self._assemble_image_data(self._read_well_info(well_id))
            self.data_well_id = well_id
        return self._extract_site(field_id)

    def get_image_window(self, well_id=None, field_id=None, data=None):
        # Assume data is not RGB(A) & uint8
        window_scanner = WindowScanner()
        window_scanner.process(data, self.get_dim_order())
        return window_scanner.get_window()

    def get_name(self):
        """
        Gets the experiment or file name.

        Returns:
            str: Name.
        """
        name = self.metadata.get('Name')
        if not name:
            name = splitall(os.path.splitext(self.uri)[0])[-2]
        return name

    def get_rows(self):
        """
        Returns the list of row identifiers.

        Returns:
            list: Row identifiers.
        """
        return self.metadata['well_info']['rows']

    def get_columns(self):
        """
        Returns the list of column identifiers.

        Returns:
            list: Column identifiers.
        """
        return self.metadata['well_info']['columns']

    def get_wells(self):
        """
        Returns the list of well identifiers.

        Returns:
            list: Well identifiers.
        """
        return list(self.metadata['wells'])

    def get_time_points(self):
        """
        Returns the list of time points.

        Returns:
            list: Time point IDs.
        """
        return self.metadata['time_points']

    def get_fields(self):
        """
        Returns the list of field indices.

        Returns:
            list: Field indices.
        """
        return self.metadata['well_info']['fields']

    def get_dim_order(self):
        """
        Returns the dimension order string.

        Returns:
            str: Dimension order.
        """
        return self.dim_order

    def get_dtype(self):
        """
        Returns the numpy dtype of the image data.

        Returns:
            dtype: Numpy dtype.
        """
        return self.dtype

    def get_pixel_size_um(self):
        """
        Returns the pixel size in micrometers.

        Returns:
            dict: Pixel size for x and y.
        """
        pixel_size = self.metadata['well_info'].get('PixelSizeUm', 1)
        return {'x': pixel_size, 'y': pixel_size}

    def get_position_um(self, well_id=None):
        """
        Returns the position in micrometers for a well.

        Args:
            well_id (str): Well identifier.

        Returns:
            dict: Position in micrometers.
        """
        well = self.metadata['wells'][well_id]
        well_info = self.metadata['well_info']
        x = well.get('CoordX', 0) * well_info['max_sizex_um']
        y = well.get('CoordY', 0) * well_info['max_sizey_um']
        return {'x': x, 'y': y}

    def get_channels(self):
        """
        Returns channel metadata.

        Returns:
            list: List of channel dicts.
        """
        channels = []
        for channel0 in self.metadata['channels']:
            channel = {}
            if 'Dye' in channel0 and channel0['Dye']:
                channel['label'] = channel0['Dye']
            if 'Color' in channel0:
                channel['color'] = hexrgb_to_rgba(channel0['Color'].lstrip('#'))
            channels.append(channel)
        return channels

    def get_nchannels(self):
        """
        Returns the number of channels.

        Returns:
            int: Number of channels.
        """
        return max(self.metadata['num_channels'], 1)

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        """
        return False

    def get_acquisitions(self):
        """
        Returns acquisition metadata.

        Returns:
            list: List of acquisition dicts.
        """
        acquisitions = []
        for index, acq in enumerate(self.metadata.get('acquisitions', [])):
            acquisitions.append({
                'id': index,
                'name': acq['Name'],
                'description': acq['Description'],
                'date_created': acq['DateCreated'].isoformat(),
                'date_modified': acq['DateModified'].isoformat()
            })
        return acquisitions

    def get_total_data_size(self):
        """
        Returns the estimated total data size.

        Returns:
            int: Total data size in bytes.
        """
        return self.metadata['max_data_size']

    def print_well_matrix(self):
        """
        Returns a string representation of the well matrix.

        Returns:
            str: Well matrix.
        """
        s = ''

        well_info = self.metadata['well_info']
        rows, cols = well_info['rows'], well_info['columns']
        used_wells = [well for well in self.metadata['wells']]

        well_matrix = []
        for row_id in rows:
            row = ''
            for col_id in cols:
                well_id = f'{row_id}{col_id}'
                row += '+' if well_id in used_wells else ' '
            well_matrix.append(row)

        header = ' '.join([pad_leading_zero(col) for col in cols])
        s += ' ' + header + '\n'
        for idx, row in enumerate(well_matrix):
            s += f'{rows[idx]} ' + '  '.join(row) + '\n'
        return s

    def print_timepoint_well_matrix(self):
        """
        Returns a string representation of the timepoint-well matrix.

        Returns:
            str: Timepoint-well matrix.
        """
        s = ''

        time_points = self.metadata['time_points']
        wells = [well for well in self.metadata['wells']]

        well_matrix = []
        for timepoint in time_points:
            wells_at_timepoint = self.db.fetch_all('''
                SELECT DISTINCT Well.Name FROM SourceImageBase
                JOIN Well ON SourceImageBase.ZoneIndex = Well.ZoneIndex
                WHERE TimeSeriesElementId = ?
            ''', (timepoint,), return_dicts=False)

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
