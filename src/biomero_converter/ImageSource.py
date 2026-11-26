from abc import ABC
from imaging_metadata_converter import convert_metadata
from imaging_metadata_converter.spatial import spatial_metadata
import numpy as np

from biomero_converter.parameters import TILE_SIZE
from biomero_converter.util import ensure_list, pad_leading_zero, parse_datetime, to_plain_types, get_level_from_scale


class ImageSource(ABC):
    """
    Abstract base class for image sources.
    """

    def __init__(self, uri, metadata={}):
        """
        Initialize ImageSource.

        Args:
            uri (str): Path to the image source.
            metadata (dict): Optional metadata dictionary.
        """
        self.uri = uri
        self.metadata = metadata

    def init_metadata(self):
        """
        Initialize and load metadata.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'init_metadata' method must be implemented by subclasses.")

    def is_screen(self):
        """
        Check if the source is a screen (multi-well).
        """
        return False

    def get_shape(self):
        """
        Get the shape of the image data.
        """
        return self.shape

    def get_shapes(self):
        """
        Get a list of shapes corresponding to the image data levels.
        """
        return self.shapes

    def get_scales(self):
        """
        Get the list of image scales.
        """
        return self.scales

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        """
        Get image data (numpy) for a well and field.

        A source implements get_data() (reading all into memory) and/or get_data_as_dask() (lazy reading, also for
        data larger than memory); by default each is taken from the other.

        Args:
            dim_order: Dimension order of data
            level (int, optional): Image resolution level
            well_id (str, optional): Well identifier
            field_id (int, optional): Field identifier
            kwargs (optional): Format specific keyword arguments.
        """
        if not self.is_lazy():
            raise NotImplementedError("The 'get_data' or 'get_data_as_dask' method must be implemented by subclasses.")
        data = self.get_data_as_dask(dim_order, level=level, well_id=well_id, field_id=field_id, **kwargs)
        return np.asarray(data)

    def get_data_as_dask(self, dim_order, level=0, **kwargs):
        """
        Get image data as dask array: lazily read if the source implements it (is_lazy()), else read in memory.

        Args:
            dim_order: Dimension order of data
            level (int, optional): Image resolution level
            kwargs (optional): Format specific keyword arguments.
        """
        return self.get_data(dim_order, level=level, **kwargs)

    def is_lazy(self):
        """
        Check if the source reads lazily (implements get_data_as_dask()): data larger than memory can be converted.
        """
        return type(self).get_data_as_dask is not ImageSource.get_data_as_dask

    def get_data_as_generator(self, dim_order, **kwargs):
        """
        Get image data as a generator function data_generator(scale) for OmeTiffWriter, yielding the tiles
        (TILE_SIZE, yx or yxc with RGB samples last) of each plane, read a row of tiles at a time from the best
        resolution level and resized to the scale.

        Args:
            dim_order: Dimension order of data (planes: the leading dimensions, before yx)
            kwargs (optional): Format specific keyword arguments.
        """
        import skimage.transform as sk_transform

        tile_dims = 'yxc' if dim_order.endswith('c') else 'yx'
        nplane_dims = len(dim_order) - len(tile_dims)

        def data_generator(scale=1):
            level, rescale = get_level_from_scale(self.get_scales(), scale)
            data = self.get_data_as_dask(dim_order, level=level, **kwargs)
            read_size = int(TILE_SIZE / rescale)
            for plane_index in np.ndindex(*data.shape[:nplane_dims]):
                plane = data[plane_index]
                for y in range(0, plane.shape[0], read_size):
                    row = np.asarray(plane[y:y + read_size])
                    for x in range(0, row.shape[1], read_size):
                        tile = row[:, x:x + read_size]
                        if rescale != 1:
                            shape = np.multiply(tile.shape[:2], rescale).astype(int)
                            shape = tuple(shape) + tile.shape[2:]
                            tile = sk_transform.resize(tile, shape, preserve_range=True).astype(tile.dtype)
                        yield tile
        return data_generator

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
        if self.get_dtype() != np.uint8:
            if data is None:
                for level, shape in enumerate(self.get_shapes()):
                    if np.prod(shape) * self.get_dtype().itemsize < 1e8:  # less than 100 MB
                        data = self.get_data(self.get_dim_order(), well_id=well_id, field_id=field_id, level=level)
                        break
            if data is not None:
                window_scanner.process(data, self.get_dim_order())
        return window_scanner.get_window()

    def get_fill_value(self):
        """
        Get the value of pixels outside the acquired area (e.g. a slide's background), or None: written as the
        zarr fill value, so chunks of only that value (empty areas) are not written.
        """
        return None

    def get_name(self):
        """
        Get the name of the image source.
        """
        return self.name

    def get_dim_order(self):
        """
        Get the dimension order string.
        """
        return self.dim_order

    def get_dtype(self):
        """
        Get the numpy dtype of the image data.
        """
        return self.dtype

    def get_pixel_size_um(self):
        """
        Get the pixel size in micrometers.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_pixel_size_um' method must be implemented by subclasses.")

    def get_position_um(self, well_id=None):
        """
        Get the position in micrometers for a well.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_position_um' method must be implemented by subclasses.")

    def get_channels(self):
        """
        Get channel metadata in NGFF format, color provided as RGBA list with values between 0 and 1
        e.g. white = [1, 1, 1, 1]

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_channels' method must be implemented by subclasses.")

    def get_nchannels(self):
        """
        Get the number of channels.
        """
        return self.nchannels

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        """
        raise NotImplementedError("The 'is_rgb' method must be implemented by subclasses.")

    def get_rows(self):
        """
        Get the list of row identifiers.
        """
        return []

    def get_columns(self):
        """
        Get the list of column identifiers.
        """
        return []

    def get_wells(self):
        """
        Get the list of well identifiers.
        """
        return []

    def get_time_points(self):
        """
        Get the list of time points.
        """
        return []

    def get_fields(self):
        """
        Get the list of field indices.
        """
        return []

    def get_acquisitions(self):
        """
        Get acquisition metadata.
        """
        return []

    def get_acquisition_datetime(self):
        """
        Get the acquisition datetime.
        """
        return self.acquisition_datetime

    def get_acquisition_end_datetime(self):
        """
        Get the datetime the acquisition ended, when its last image was collected.

        Returns:
            datetime: End datetime, or None if the metadata does not give it.
        """
        return None

    def get_significant_bits(self):
        """
        Get the number of significant bits in the image data.
        """
        return self.bits_per_pixel

    def get_metadata(self):
        """
        Get the metadata dictionary.

        Returns:
            dict: Metadata dictionary.
        """
        return self.metadata

    def get_acquisition_metadata(self):
        """
        Get microscope information. This can include details such as the microscope model, objective lens, and other relevant information.
        """
        return {}

    def get_source_metadata(self):
        """
        Get all of the source's metadata, but what only lays out or encodes the pixel data, for the imaging
        metadata converter: by default the acquisition metadata.
        """
        return self.get_acquisition_metadata()

    def get_model_metadata(self):
        """
        Get the source metadata mapped onto the common imaging metadata model
        (https://github.com/NL-BioImaging/imaging-metadata-converter). Unmapped fields keep their original path.

        Returns:
            dict: Nested model metadata, e.g. {'Instrument': {'Manufacturer': ..., 'Model': ...}, ...}.
        """
        # converted once, as the OME writer and the RO-Crate both ask for it; as plain data (bytes as text, numpy
        # values as numbers), as the converter's JSON examples hold it
        if getattr(self, '_model_metadata', None) is None:
            self._model_metadata = convert_metadata(to_plain_types(self.get_source_metadata()))
        return self._model_metadata

    def get_spatial_metadata(self, shape=None):
        """
        Get the pixel size and position (µm) and the image's and stage's rotation (degrees) the model metadata gives,
        as every reader of the imaging metadata converter reads them (its spatial_metadata).

        Args:
            shape (dict, optional): {axis: pixels} of the image, for a pixel size given as the field of view.

        Returns:
            dict: {'pixel_size': {axis: µm}, 'position': {axis: µm}, 'rotation': degrees or None,
            'stage_rotation': degrees or None}.
        """
        return spatial_metadata(self.get_model_metadata(), shape)

    def get_rotation(self):
        """
        Get how far the image is turned against the stage axes, in degrees (its scan rotation).

        Returns:
            float: Rotation in degrees, or None if the metadata does not give it.
        """
        return getattr(self, 'rotation', None)

    def get_model_acquisition_datetime(self):
        """
        Get the acquisition datetime the model metadata gives (Image.AcquisitionDate), from the vendor metadata.

        Returns:
            datetime: Acquisition datetime, or None if the metadata does not give it.
        """
        images = ensure_list(self.get_model_metadata().get('Image') or [])
        if images and isinstance(images[0], dict):
            return parse_datetime(images[0].get('AcquisitionDate'))
        return None

    def get_total_data_size(self):
        """
        Get the estimated total data size.

        Returns:
            int: Total data size in bytes.
        """
        image_size = np.prod(self.get_shape()) * np.dtype(self.get_dtype()).itemsize
        if self.is_screen():
            nwells = len(self.get_wells())
            nfields = len(self.get_fields())
            total_size = image_size * nwells * nfields
        else:
            total_size = image_size
        return total_size

    def close(self):
        """
        Close the image source.
        """
        pass

    def print_well_matrix(self):
        """
        Print a matrix representation of the well plate.
        """
        s = ''

        rows, cols = self.get_rows(), self.get_columns()
        used_wells = [well for well in self.get_wells()]

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
