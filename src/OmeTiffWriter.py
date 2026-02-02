import inspect
import numpy as np
import os.path
from skimage.transform import resize
from tifffile import TiffWriter

from src.ome_tiff_util import create_metadata, create_binaryonly_metadata, create_resolution_metadata, create_uuid
from src.OmeWriter import OmeWriter
from src.parameters import *
from src.util import *
from src.WindowScanner import WindowScanner


class OmeTiffWriter(OmeWriter):
    """
    Writes image data and metadata to OME-TIFF files.
    """
    def __init__(self, verbose=False):
        """
        Initialize OmeTiffWriter.

        Args:
            verbose (bool): If True, prints progress info.
        """
        super().__init__()
        self.verbose = verbose

    def write(self, filepath, source, **kwargs):
        """
        Writes image or screen data to OME-TIFF files.

        Args:
            filepath (str): Output file path.
            source (ImageSource): Source object.
            **kwargs: Additional options (e.g. wells selection).

        Returns:
            dict: Containing output_path: str or list Output file path(s) and data window.
        """

        dim_order = ''
        source_dim_order = source.get_dim_order()
        if source.get_time_points():
            dim_order += 't'
        if 'c' in source_dim_order and not source.is_rgb():
            dim_order += 'c'
        if 'z' in source_dim_order:
            dim_order += 'z'
        dim_order += 'yx'
        if 'c' in source_dim_order and source.is_rgb():
            dim_order += 'c'
        self.dim_order = dim_order

        if source.is_screen():
            filepath, total_size, window = self._write_screen(filepath, source, **kwargs)
        else:
            filepath, total_size, window = self._write_image(filepath, source, **kwargs)

        if self.verbose:
            print(f'Total data written: {print_hbytes(total_size)}')

        return {'output_path': filepath, 'window': window}

    def _write_screen(self, filename, source, **kwargs):
        """
        Writes multi-well screen data to separate TIFF files and companion metadata.

        Args:
            filename (str): Output file name.
            source (ImageSource): Source object.
            **kwargs: Additional options (e.g. wells selection).

        Returns:
            tuple: (List of output paths, total data size, image window)
        """
        # writes separate tiff files for each field, and separate metadata companion file
        window = []
        output_paths = []
        filepath, filename = os.path.split(filename)
        filetitle = os.path.splitext(filename)[0].rstrip('.ome')

        companion_filename = os.path.join(filepath, filetitle + '.companion.ome')
        companion_uuid = create_uuid()

        wells = kwargs.get('wells', source.get_wells())
        fields = list(map(str, source.get_fields()))

        total_size = 0
        image_uuids = []
        image_filenames = []
        for well_id in wells:
            for field_id in fields:
                resolution, resolution_unit = create_resolution_metadata(source)
                data = source.get_data(self.dim_order, well_id=well_id, field_id=field_id)

                filename = f'{filetitle}'
                filename += f'_{pad_leading_zero(well_id)}'
                if field_id is not None:
                    filename += f'_{pad_leading_zero(field_id)}'
                filename = os.path.join(filepath, filename + '.ome.tiff')
                xml_metadata, image_uuid = create_binaryonly_metadata(os.path.basename(companion_filename), companion_uuid)

                size, window = self._write_tiff(filename, source, data,
                                                resolution=resolution, resolution_unit=resolution_unit,
                                                tile_size=TILE_SIZE, compression=TIFF_COMPRESSION,
                                                xml_metadata=xml_metadata,
                                                pyramid_levels=PYRAMID_LEVELS, pyramid_downscale=PYRAMID_DOWNSCALE,
                                                well_id=well_id, field_id=field_id, **kwargs)

                image_uuids.append(image_uuid)
                image_filenames.append(os.path.basename(filename))
                output_paths.append(filename)
                total_size += size

        xml_metadata = create_metadata(source,
                                       uuid=companion_uuid, image_uuids=image_uuids, image_filenames=image_filenames,
                                       wells=wells)
        with open(companion_filename, 'wb') as file:
            file.write(xml_metadata.encode())

        output_paths = [companion_filename] + output_paths

        return output_paths, total_size, window

    def _write_image(self, filename, source, **kwargs):
        """
        Writes single image data to a TIFF file.

        Args:
            filename (str): Output file name.
            source (ImageSource): Source object.
            **kwargs: Additional options.

        Returns:
            tuple: (Output path, data size)
        """
        xml_metadata = create_metadata(source, image_filenames=[filename])
        resolution, resolution_unit = create_resolution_metadata(source)
        data = source.get_data_as_generator(self.dim_order)

        size, window = self._write_tiff(filename, source, data,
                                        resolution=resolution, resolution_unit=resolution_unit,
                                        tile_size=TILE_SIZE, compression=TIFF_COMPRESSION,
                                        xml_metadata=xml_metadata,
                                        pyramid_levels=PYRAMID_LEVELS, pyramid_downscale=PYRAMID_DOWNSCALE,
                                        **kwargs)

        return filename, size, window

    def _write_tiff(self, filename, source, data,
                    resolution=None, resolution_unit=None, tile_size=None, compression=None, compressionargs=None,
                    xml_metadata=None, pyramid_levels=0, pyramid_downscale=2, well_id=None, field_id=None, **kwargs):
        """
        Writes image data to a TIFF file with optional pyramids and metadata.

        Args:
            filename (str): Output file name.
            source (ImageSource): Source object.
            data (ndarray or generator): Image data.
            resolution (tuple, optional): Pixel resolution.
            resolution_unit (str, optional): Resolution unit.
            tile_size (int or tuple, optional): Tile size.
            compression (str, optional): Compression type.
            xml_metadata (str, optional): OME-XML metadata.
            pyramid_levels (int): Number of pyramid levels.
            pyramid_downscale (int): Pyramid downscale factor.

        Returns:
            int: Data size in bytes.
        """
        is_generator = inspect.isgeneratorfunction(data)
        if is_generator:
            data_generator = data
            shape = list(source.shape)
            dtype = source.get_dtype()
        else:
            shape = list(data.shape)
            dtype = data.dtype

        x_index = self.dim_order.index('x')
        y_index = self.dim_order.index('y')
        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = [tile_size] * 2
            if tile_size[0] > shape[y_index] or tile_size[1] > shape[x_index]:
                tile_size = None

        if xml_metadata is not None:
            # set ome=False to provide custom OME xml in description
            xml_metadata_bytes = xml_metadata.encode()
            is_ome = False
        else:
            xml_metadata_bytes = None
            is_ome = True

        # maximum size (w/o compression)
        if is_generator:
            data_size = np.prod(shape) * dtype.itemsize
        else:
            data_size = data.size * data.itemsize
        max_size = 0
        scale = 1
        for level in range(1 + pyramid_levels):
            max_size += data_size * scale ** 2
            scale /= pyramid_downscale
        is_bigtiff = (max_size > 2 ** 32)

        window_scanner = WindowScanner()
        with TiffWriter(filename, bigtiff=is_bigtiff, ome=is_ome) as writer:
            for level in range(pyramid_levels + 1):
                if level == 0:
                    scale = 1
                    subifds = pyramid_levels
                    subfiletype = None
                    new_shape = shape
                else:
                    scale /= pyramid_downscale
                    new_shape = list(shape)
                    new_shape[x_index] = int(shape[x_index] * scale)
                    new_shape[y_index] = int(shape[y_index] * scale)
                    if not is_generator:
                        data = resize(data, new_shape, preserve_range=True).astype(dtype)
                    subifds = None
                    subfiletype = 1
                    xml_metadata_bytes = None
                if is_generator:
                    data = data_generator(scale)
                writer.write(data, shape=tuple(new_shape), dtype=dtype, metadata={'axes': self.dim_order},
                             subifds=subifds, subfiletype=subfiletype,
                             resolution=resolution, resolutionunit=resolution_unit, tile=tile_size,
                             compression=compression, compressionargs=compressionargs,
                             description=xml_metadata_bytes)
                if level == pyramid_levels:
                    window = source.get_image_window(window_scanner, well_id=well_id, field_id=field_id, data=data)
        return data_size, window
