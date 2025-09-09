import numpy as np
import os.path
from skimage.transform import resize
from tifffile import TiffWriter

from src.ome_tiff_util import create_metadata, create_binaryonly_metadata, create_resolution_metadata, create_uuid
from src.OmeWriter import OmeWriter
from src.util import *


class OmeTiffWriter(OmeWriter):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def write(self, filepath, source, **kwargs):
        if source.is_screen():
            filepath, total_size = self._write_screen(filepath, source, **kwargs)
        else:
            filepath, total_size = self._write_image(filepath, source, **kwargs)

        if self.verbose:
            print(f'Total data written: {print_hbytes(total_size)}')

        return filepath

    def _write_screen(self, filename, source, **kwargs):
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
                                        tile_size=kwargs.get('tile_size'), compression=kwargs.get('compression'),
                                        xml_metadata=xml_metadata, pyramid_levels=4)

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
        xml_metadata, _ = create_metadata(source)
        resolution, resolution_unit = create_resolution_metadata(source)
        data = source.get_data()

        size = self._write_tiff(filename, source, data,
                                resolution=resolution, resolution_unit=resolution_unit,
                                tile_size=kwargs.get('tile_size'), compression=kwargs.get('compression'),
                                xml_metadata=xml_metadata, pyramid_levels=4)

        return filename, size

    def _write_tiff(self, filename, source, data,
                  resolution=None, resolution_unit=None, tile_size=None, compression=None,
                  xml_metadata=None, pyramid_levels=0, pyramid_scale=2):

        dim_order = source.get_dim_order()
        shape = data.shape
        x_index = dim_order.index('x')
        y_index = dim_order.index('y')
        size = shape[x_index], shape[y_index]

        if tile_size is not None and isinstance(tile_size, int):
            tile_size = [tile_size] * 2

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
        max_size = data.size * data.itemsize
        base_size = np.divide(max_size, np.prod(size))
        scale = 1
        for level in range(pyramid_levels):
            max_size += np.prod(size) * scale * base_size
            scale /= pyramid_scale
        bigtiff = (max_size > 2 ** 32)

        size = data.size * data.itemsize
        with TiffWriter(filename, bigtiff=bigtiff, ome=is_ome) as writer:
            for level in range(pyramid_levels + 1):
                if level == 0:
                    scale = 1
                    subifds = pyramid_levels
                    subfiletype = None
                else:
                    scale /= pyramid_scale
                    new_shape = list(shape)
                    new_shape[x_index] = int(shape[x_index] * scale)
                    new_shape[y_index] = int(shape[y_index] * scale)
                    data = resize(data, new_shape)
                    subifds = None
                    subfiletype = 1
                    xml_metadata_bytes = None
                writer.write(data, subifds=subifds, subfiletype=subfiletype,
                             resolution=resolution, resolutionunit=resolution_unit, tile=tile_size,
                             compression=compression, description=xml_metadata_bytes)
        return size
