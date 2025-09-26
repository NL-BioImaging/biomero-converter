import numpy as np
import os.path
from skimage.transform import resize
from tifffile import TiffWriter

from src.ome_tiff_util import create_metadata, create_binaryonly_metadata, create_resolution_metadata, create_uuid
from src.OmeWriter import OmeWriter
from src.parameters import *
from src.util import *


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
            **kwargs: Additional options.

        Returns:
            str or list: Output file path(s).
        """
        if source.is_screen():
            filepath, total_size = self._write_screen(filepath, source, **kwargs)
        else:
            filepath, total_size = self._write_image(filepath, source, **kwargs)

        if self.verbose:
            print(f'Total data written: {print_hbytes(total_size)}')

        return filepath

    def _write_screen(self, filename, source, **kwargs):
        """
        Writes multi-well screen data to separate TIFF files and companion metadata.

        Args:
            filename (str): Output file name.
            source (ImageSource): Source object.
            **kwargs: Additional options.

        Returns:
            tuple: (List of output paths, total data size)
        """
        # writes separate tiff files for each field, and separate metadata companion file
        output_paths = []
        filepath, filename = os.path.split(filename)
        filetitle = os.path.splitext(filename)[0].rstrip('.ome')

        companion_filename = os.path.join(filepath, filetitle + '.companion.ome')
        companion_uuid = create_uuid()

        total_size = 0
        image_uuids = []
        image_filenames = []
        for well_id in source.get_wells():
            for field in source.get_fields():
                resolution, resolution_unit = create_resolution_metadata(source)
                data = source.get_data(well_id, field)

                filename = f'{filetitle}'
                filename += f'_{pad_leading_zero(well_id)}'
                if field is not None:
                    filename += f'_{pad_leading_zero(field)}'
                filename = os.path.join(filepath, filename + '.ome.tiff')
                xml_metadata, image_uuid = create_binaryonly_metadata(os.path.basename(companion_filename), companion_uuid)

                size = self._write_tiff(filename, source, data,
                                        resolution=resolution, resolution_unit=resolution_unit,
                                        tile_size=TILE_SIZE, compression=TIFF_COMPRESSION,
                                        xml_metadata=xml_metadata,
                                        pyramid_levels=PYRAMID_LEVELS, pyramid_downscale=PYRAMID_DOWNSCALE)

                image_uuids.append(image_uuid)
                image_filenames.append(os.path.basename(filename))
                output_paths.append(filename)
                total_size += size

        xml_metadata = create_metadata(source, companion_uuid, image_uuids, image_filenames)
        with open(companion_filename, 'wb') as file:
            file.write(xml_metadata.encode())

        output_paths = [companion_filename] + output_paths
        return output_paths, total_size

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
        xml_metadata = create_metadata(source)
        resolution, resolution_unit = create_resolution_metadata(source)
        data = source.get_data()

        size = self._write_tiff(filename, source, data,
                                resolution=resolution, resolution_unit=resolution_unit,
                                tile_size=TILE_SIZE, compression=TIFF_COMPRESSION,
                                xml_metadata=xml_metadata,
                                pyramid_levels=PYRAMID_LEVELS, pyramid_downscale=PYRAMID_DOWNSCALE)

        return filename, size

    def _write_tiff(self, filename, source, data,
                    resolution=None, resolution_unit=None, tile_size=None, compression=None, compressionargs=None,
                    xml_metadata=None, pyramid_levels=0, pyramid_downscale=2):
        """
        Writes image data to a TIFF file with optional pyramids and metadata.

        Args:
            filename (str): Output file name.
            source (ImageSource): Source object.
            data (ndarray): Image data.
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
        dim_order = source.get_dim_order()
        shape = list(data.shape)
        if source.is_rgb() and source.get_nchannels() in (3, 4) and dim_order[-1] != 'c':
            old_dimc = dim_order.index('c')
            data = np.moveaxis(data, old_dimc, -1)
            dim_order = dim_order[:old_dimc] + dim_order[old_dimc+1:] + 'c'
            shape = shape[:old_dimc] + shape[old_dimc+1:] + [shape[old_dimc]]

        x_index = dim_order.index('x')
        y_index = dim_order.index('y')
        source_type = source.get_dtype()
        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = [tile_size] * 2
            if tile_size[0] > shape[y_index] or tile_size[1] > shape[x_index]:
                tile_size = None

        if resolution is not None:
            # tifffile only supports x/y pyramid resolution
            resolution = tuple(resolution[0:2])

        if xml_metadata is not None:
            # set ome=False to provide custom OME xml in description
            xml_metadata_bytes = xml_metadata.encode()
            is_ome = False
        else:
            xml_metadata_bytes = None
            is_ome = True

        # maximum size (w/o compression)
        data_size = data.size * data.itemsize
        max_size = 0
        scale = 1
        for level in range(1 + pyramid_levels):
            max_size += data_size * scale ** 2
            scale /= pyramid_downscale
        bigtiff = (max_size > 2 ** 32)

        with TiffWriter(filename, bigtiff=bigtiff, ome=is_ome) as writer:
            for level in range(pyramid_levels + 1):
                if level == 0:
                    scale = 1
                    subifds = pyramid_levels
                    subfiletype = None
                else:
                    scale /= pyramid_downscale
                    new_shape = list(shape)
                    new_shape[x_index] = int(shape[x_index] * scale)
                    new_shape[y_index] = int(shape[y_index] * scale)
                    data = resize(data, new_shape, preserve_range=True).astype(source_type)
                    subifds = None
                    subfiletype = 1
                    xml_metadata_bytes = None
                writer.write(data, subifds=subifds, subfiletype=subfiletype,
                             resolution=resolution, resolutionunit=resolution_unit, tile=tile_size,
                             compression=compression, compressionargs=compressionargs,
                             description=xml_metadata_bytes)
        return data_size
