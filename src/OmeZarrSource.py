from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import zarr

from src.ImageSource import ImageSource
from src.ome_zarr_util import *
from src.parameters import *
from src.util import split_well_name, print_hbytes


class OmeZarrSource(ImageSource):
    def init_metadata(self):
        """
        Initializes and loads metadata from the (OME) TIFF file.

        Returns:
            dict: Metadata dictionary.
        """
        location = parse_url(self.uri)
        if location is None:
            raise FileNotFoundError(f'Error parsing ome-zarr file {self.uri}')
        reader = Reader(location)
        # nodes may include images, labels etc
        nodes = list(reader())
        # first node will be the image pixel data
        if len(nodes) == 0 and 'bioformats2raw.layout' in location.root_attrs:
            # try to read bioformats2raw format
            # TODO: use paths provided in metadata
            reader = Reader(parse_url(self.uri + '/0'))
            nodes = list(reader())
            if len(nodes) == 0:
                raise FileNotFoundError(f'No image data found in ome-zarr file {self.uri}')
        image_node = nodes[0]
        self.data = image_node.data
        self.shape = self.data[0].shape
        self.dtype = self.data[0].dtype

        self.metadata = image_node.metadata
        # channel metadata from ome-zarr-py limited; get from root_attrs manually
        self.root_metadata = reader.zarr.root_attrs

        axes = self.metadata.get('axes', [])
        self.dim_order = ''.join([axis.get('name') for axis in axes])
        self.plate = self.metadata.get('metadata', {}).get('plate')
        self.is_plate = self.plate is not None

        pixel_sizes0 = [transform for transform
                        in self.metadata['coordinateTransformations'][0]
                        if transform['type'] == 'scale'][0]['scale']
        self.pixel_size = {axis: pixel_size for axis, pixel_size in zip(self.dim_order, pixel_sizes0) if axis in 'xyz'}
        if self.is_plate:
            self.name = self.plate.get('name', '')
            self.wells = [well['path'].replace('/', '') for well in self.plate.get('wells')]
            self.fields = list(range(self.plate.get('field_count', 0)))
            paths = []
            for well in self.plate.get('wells'):
                path = well['path']
                if self.fields:
                    paths.extend([f'{path}/{field}' for field in self.fields])
                else:
                    paths.append(path)
            # TODO: read paths
        else:
            self.name = self.metadata.get('name', '')

    def is_screen(self):
        """
        Checks if the source is a plate/screen.

        Returns:
            bool: True if plate/screen.
        """
        return self.is_plate

    def get_shape(self):
        """
        Returns the shape of the image data.

        Returns:
            tuple: Shape of the image data.
        """
        return self.shape

    def get_data(self, well_id=None, field_id=None, level=0, **kwargs):
        """
        Gets image data from ZARR nodes.

        Returns:
            ndarray: Image data.
        """

        return self.data[level]

    def get_image_window(self, well_id=None, field_id=None, data=None):

        return [], []


    def get_name(self):
        """
        Gets the image or plate name.

        Returns:
            str: Name.
        """
        return self.name

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
            dict: Pixel size for x, y, (and z).
        """
        return self.pixel_size

    def get_position_um(self, well_id=None):
        """
        Returns the position in micrometers.

        Returns:
            dict: Position in micrometers.
        """
        return self.position

    def get_channels(self):
        """
        Returns channel metadata.

        Returns:
            list: List of channel dicts.
        """
        return self.channels

    def get_nchannels(self):
        """
        Returns the number of channels.

        Returns:
            int: Number of channels.
        """
        return self.shape[self.dim_order.index('c')] if 'c' in self.dim_order else 1

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        """
        return self.get_nchannels() == 3

    def get_rows(self):
        """
        Returns the list of row identifiers.

        Returns:
            list: Row identifiers.
        """
        return self.rows

    def get_columns(self):
        """
        Returns the list of column identifiers.

        Returns:
            list: Column identifiers.
        """
        return self.columns

    def get_wells(self):
        """
        Returns the list of well identifiers.

        Returns:
            list: Well identifiers.
        """
        return self.wells

    def get_time_points(self):
        """
        Returns the list of time points.

        Returns:
            list: Time point IDs.
        """
        nt = 1
        if 't' in self.dim_order:
            t_index = self.dim_order.index('t')
            nt = self.tiff.pages.first.shape[t_index]
        return list(range(nt))

    def get_fields(self):
        """
        Returns the list of field indices.

        Returns:
            list: Field indices.
        """
        return self.fields

    def get_acquisitions(self):
        """
        Returns acquisition metadata (empty for TIFF).

        Returns:
            list: Empty list.
        """
        return self.acquisitions

    def get_total_data_size(self):
        """
        Returns the estimated total data size.

        Returns:
            int: Total data size in bytes.
        """
        total_size = np.prod(self.shape)
        if self.is_plate:
            total_size *= len(self.get_wells()) * len(self.get_fields())
        return total_size
