import dask.array as da
from datetime import datetime
import dateutil
from enum import Enum
import numpy as np
import os.path
from tifffile import TiffFile, imread, PHOTOMETRIC

from src.ImageSource import ImageSource
from src.ome_tiff_util import metadata_to_dict, read_ome_xml_metadata
from src.parameters import TILE_SIZE
from src.tiff_metadata import PRIVATE_TAG_CODE, get_extra_metadata, get_pixel_size_um, get_position_um
from src.util import convert_to_um, ensure_list, redimension_data, get_filetitle, without_keys


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
        self.is_ome = False
        if 'tif' in ext:
            image_filename = uri
        elif ext in ('.ome', '.xml'):
            # read metadata
            with open(uri, 'rb') as file:
                self.metadata = metadata_to_dict(file.read().decode())
            self.is_ome = True
            # try to open linked ome-tiff file
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

        if image_filename:
            self.tiff = TiffFile(image_filename)
        else:
            self.tiff = None

    def init_metadata(self):
        acquisition_datetime = None
        pixel_size = {}
        position = {}
        rotation = None
        channels = []
        acquisition_metadata = {}
        wells = {}
        rows = []
        columns = []
        fields = []
        image_refs = {}
        metadata = {}

        if self.tiff:
            self.is_ome = self.tiff.is_ome
            self.is_imagej = self.tiff.is_imagej

            if self.tiff.series:
                page = self.tiff.series[0]
            else:
                page = self.tiff.pages.first
            # pyramid levels of the first series only (other series are e.g. thumbnails), as read in get_data()
            pages = page.levels if hasattr(page, 'levels') else [page]
            self.shapes = [page.shape for page in pages]
            self.shape = page.shape
            self.dim_order = page.axes.lower().replace('s', 'c').replace('r', '')
            x_index, y_index = self.dim_order.index('x'), self.dim_order.index('y')
            self.scales = [float(np.mean([shape[x_index] / self.shape[x_index], shape[y_index] / self.shape[y_index]]))
                           for shape in self.shapes]
            self.is_photometric_rgb = (self.tiff.pages.first.photometric == PHOTOMETRIC.RGB)
            self.nchannels = self.shape[self.dim_order.index('c')] if 'c' in self.dim_order else 1

        source_metadata = {}
        if self.is_ome:
            if self.tiff:
                metadata = metadata_to_dict(self.tiff.ome_metadata)
                # the OME document whole, but where its pixel data lies and how it is written
                source_metadata = without_keys(get_extra_metadata(self.tiff), OME_LAYOUT_KEYS)
            if metadata and not 'BinaryOnly' in metadata:
                self.metadata = metadata
            (name, is_plate, pixel_size, position, dtype, bits_per_pixel, channels, acquisition_metadata, acquisition_datetime,
             wells, rows, columns, fields, image_refs) = read_ome_xml_metadata(self.metadata)
        else:
            is_plate = False
            if self.is_imagej:
                metadata = self.tiff.imagej_metadata
                pixel_size = get_fiji_pixelsize(metadata)

            metadata |= {key: value for page in self.tiff.pages for key, value in tags_to_dict(page.tags).items()
                         if key not in ('StripOffsets', 'StripByteCounts', 'TileOffsets', 'TileByteCounts', 'JPEGTables')}

            # vendor metadata in generic form, with the pixel size and position found in it
            acquisition_metadata = get_extra_metadata(self.tiff)
            # and beside it the baseline tags of the image, but those that only lay out or encode its pixels
            source_metadata = dict(acquisition_metadata)
            for tag in self.tiff.pages.first.tags.values():
                if tag.code < PRIVATE_TAG_CODE and tag.name not in TIFF_LAYOUT_TAGS:
                    source_metadata.setdefault(tag.name, plain_tag_value(tag.value))
            shape = dict(zip(self.dim_order, self.shape))
            for dim, size in get_pixel_size_um(self.tiff, acquisition_metadata, shape).items():
                pixel_size.setdefault(dim, size)
            position = get_position_um(acquisition_metadata)

            self.metadata = metadata
            name = self.tiff.filename
            if not acquisition_datetime:
                if 'DateTime' in self.metadata:
                    acquisition_datetime = parse_tiff_datetime(self.metadata['DateTime'])
                else:
                    acquisition_datetime = datetime.fromtimestamp(self.tiff.fstat.st_ctime)
            dtype = page.dtype
            bits_per_pixel = dtype.itemsize * 8

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
        self.rotation = rotation
        self.channels = channels
        self.dtype = dtype
        self.bits_per_pixel = bits_per_pixel
        self.acquisition_metadata = acquisition_metadata
        self.source_metadata = source_metadata
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
            image_id = self.image_refs[well_id][int(field_id)]
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

    def get_acquisition_metadata(self):
        return self.acquisition_metadata

    def get_source_metadata(self):
        return self.source_metadata

    def close(self):
        self.tiff.close()


def get_fiji_pixelsize(metadata):
    pixel_size = {}
    pixel_size_unit = metadata.get('unit', '').encode().decode('unicode_escape')
    if 'scales' in metadata:
        for dim, scale in zip(['x', 'y'], metadata['scales'].split(',')):
            scale = scale.strip()
            if scale != '':
                pixel_size[dim] = convert_to_um(float(scale), pixel_size_unit)
    if 'spacing' in metadata:
        pixel_size['z'] = convert_to_um(metadata['spacing'], pixel_size_unit)
    return pixel_size


# baseline TIFF tags that only lay out or encode the pixel data, meaningless once the pixels are converted
TIFF_LAYOUT_TAGS = {
    'NewSubfileType', 'SubfileType', 'Compression', 'Predictor', 'PlanarConfiguration', 'FillOrder',
    'StripOffsets', 'StripByteCounts', 'RowsPerStrip', 'TileWidth', 'TileLength', 'TileDepth', 'TileOffsets',
    'TileByteCounts', 'SubIFDs', 'JPEGTables', 'JPEGProc', 'JPEGInterchangeFormat', 'JPEGInterchangeFormatLength',
    'JPEGRestartInterval', 'JPEGLosslessPredictors', 'JPEGPointTransforms', 'JPEGQTables', 'JPEGDCTables',
    'JPEGACTables', 'YCbCrCoefficients', 'YCbCrSubSampling', 'YCbCrPositioning', 'ReferenceBlackWhite',
    'ExtraSamples', 'ColorMap'}

# OME's description of where and how its pixel data is written
OME_LAYOUT_KEYS = ('TiffData', 'BinData', 'BigEndian', 'Interleaved')


def plain_tag_value(value):
    """A tag's value as plain data: an enumeration by its name, text read as text."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


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


def parse_tiff_datetime(value):
    """
    Parses a TIFF DateTime tag value.

    Args:
        value (str): 'YYYY:MM:DD HH:MM:SS' as the TIFF spec defines, or another format some writers use.

    Returns:
        datetime: Parsed datetime.
    """
    try:
        return datetime.strptime(value.strip(), '%Y:%m:%d %H:%M:%S')
    except ValueError:
        # dateutil reads the TIFF format as a time of today, so only use it for other formats
        return dateutil.parser.parse(value)
