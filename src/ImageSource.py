from abc import ABC


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

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'is_screen' method must be implemented by subclasses.")

    def get_shape(self):
        """
        Get the shape of the image data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_shape' method must be implemented by subclasses.")

    def get_data(self, dim_order, well_id=None, field_id=None, **kwargs):
        """
        Get image data for a well and field.

        Args:
            dim_order: Dimension order of data
            well_id (str, optional): Well identifier
            field_id (int, optional): Field identifier
            kwargs (optional): Format specific keyword arguments.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_data' method must be implemented by subclasses.")

    def get_data_as_dask(self, dim_order, **kwargs):
        """
        Get image data for a well and field.

        Args:
            dim_order: Dimension order of data
            well_id (str, optional): Well identifier
            field_id (int, optional): Field identifier
            kwargs (optional): Format specific keyword arguments.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_data' method must be implemented by subclasses.")

    def get_image_window(self, window_scanner, well_id=None, field_id=None, data=None):
        """
        Get image value range window (for a well & field or from provided data).

        Args:
            window_scanner (WindowScanner): WindowScanner object to compute window.
            well_id (str, optional): Well identifier
            field_id (int, optional): Field identifier
            data (ndarray, optional): Image data to compute window from.
        """
        return window_scanner.get_window()

    def get_name(self):
        """
        Get the name of the image source.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_name' method must be implemented by subclasses.")

    def get_dim_order(self):
        """
        Get the dimension order string.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_dim_order' method must be implemented by subclasses.")

    def get_dtype(self):
        """
        Get the numpy dtype of the image data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_dtype' method must be implemented by subclasses.")

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

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_nchannels' method must be implemented by subclasses.")

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        """
        raise NotImplementedError("The 'is_rgb' method must be implemented by subclasses.")

    def get_rows(self):
        """
        Get the list of row identifiers.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_rows' method must be implemented by subclasses.")

    def get_columns(self):
        """
        Get the list of column identifiers.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_columns' method must be implemented by subclasses.")

    def get_wells(self):
        """
        Get the list of well identifiers.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_wells' method must be implemented by subclasses.")

    def get_time_points(self):
        """
        Get the list of time points.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_time_points' method must be implemented by subclasses.")

    def get_fields(self):
        """
        Get the list of field indices.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_fields' method must be implemented by subclasses.")

    def get_acquisitions(self):
        """
        Get acquisition metadata.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_acquisitions' method must be implemented by subclasses.")

    def get_total_data_size(self):
        """
        Get the estimated total data size.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'get_total_data_size' method must be implemented by subclasses.")

    def close(self):
        """
        Close the image source.
        """
        pass
