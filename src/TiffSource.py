import dask
import dask.array as da
from enum import Enum
import numpy as np
import os.path
from tifffile import TiffFile, imread, PHOTOMETRIC

from src.ImageSource import ImageSource
from src.color_conversion import int_to_rgba
from src.ome_tiff_util import metadata_to_dict, create_col_row_label
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
        """
        Initializes and loads metadata from the (OME) TIFF file.

        Returns:
            dict: Metadata dictionary.
        """
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
        self.is_photometric_rgb = (self.tiff.pages.first.photometric == PHOTOMETRIC.RGB)

        if self.is_ome:
            metadata = metadata_to_dict(self.tiff.ome_metadata)
            if metadata and not 'BinaryOnly' in metadata:
                self.metadata = metadata
            image0 = ensure_list(self.metadata.get('Image', []))[0]
            self.is_plate = 'Plate' in self.metadata
            if self.is_plate:
                plate = self.metadata['Plate']
                self.name = plate.get('Name')
                rows = set()
                columns = set()
                wells = {}
                fields = []
                image_refs = {}
                for well in ensure_list(plate['Well']):
                    row = create_col_row_label(well['Row'], plate['RowNamingConvention'])
                    column = create_col_row_label(well['Column'], plate['ColumnNamingConvention'])
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
                self.name = image0.get('Name')
            if not self.name:
                self.name = get_filetitle(self.uri)
            self.name = str(self.name).rstrip('.tiff').rstrip('.tif').rstrip('.ome')
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

    def get_data(self, dim_order, well_id=None, field_id=None, **kwargs):
        if well_id is not None:
            index = self.image_refs[well_id][str(field_id)]
            tiff = TiffFile(self.image_filenames[index])
        else:
            tiff = self.tiff
        data = tiff.asarray()
        return redimension_data(data, self.dim_order, dim_order)

    def get_data_as_dask(self, dim_order, level=0, **kwargs):
        lazy_array = dask.delayed(imread)(self.uri, level=level)
        data = da.from_delayed(lazy_array, shape=self.shapes[level], dtype=self.dtype)
        data = data.rechunk(TILE_SIZE)
        return redimension_data(data, self.dim_order, dim_order)

    def get_image_window(self, window_scanner, well_id=None, field_id=None, data=None):
        """
        Get image value range window (for a well & field or from provided data).

        Args:
            window_scanner (WindowScanner): WindowScanner object to compute window.
            well_id (str, optional): Well identifier
            field_id (int, optional): Field identifier
            data (ndarray, optional): Image data to compute window from.
        """
        # For RGB(A) uint8 images don't change color value range
        if not (self.is_photometric_rgb and self.dtype == np.uint8):
            if data is None:
                if self.tiff.series:
                    page = self.tiff.series[0]
                else:
                    page = self.tiff.pages.first
                if hasattr(page, 'levels'):
                    small_page = None
                    for level_page in page.levels:
                        if level_page.nbytes < 1e8:  # less than 100 MB
                            small_page = level_page
                            break
                    if small_page:
                        data = small_page.asarray()
            if data is not None:
                window_scanner.process(data, self.source_dim_order)
        return window_scanner.get_window()


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
        if self.pixel_size:
            return self.pixel_size
        else:
            return {'x': 1, 'y': 1}

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
        return self.shape[1]

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        """
        return self.is_photometric_rgb

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
        return []

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

    def close(self):
        """
        Closes the TIFF file.
        """
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
