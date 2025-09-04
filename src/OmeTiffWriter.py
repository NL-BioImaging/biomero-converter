# TODO: requires proper implementation including ome xml metadata

import logging
import numpy as np
import os.path
from skimage.transform import resize
from tifffile import TiffWriter

from ome_tiff_util import create_metadata, create_resolution_data
from src.OmeWriter import OmeWriter
from src.util import *


class OmeTiffWriter(OmeWriter):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def write(self, filename, source, well_id=None, field_id=None, tile_size=None, compression=None, **kwargs):
        is_screen = source.is_screen()
        # TODO: if is_screen, decide whether to write single tiff with screen + metadata, or separate tiff files + separate companion file with metadata

        filepath, filename = os.path.split(filename)
        filetitle, ext = os.path.splitext(filename)
        filename = f'{filetitle}'
        filename += f'_{pad_leading_zero(well_id)}'
        if field_id is not None and field_id >= 0:
            filename += f'_{pad_leading_zero(field_id)}'
        filename = os.path.join(filepath, filename + ext)

        data = source.get_data(well_id, field_id)

        xml_metadata = create_metadata(source)

        resolution, resolution_unit = create_resolution_data(source)
        self._write_tiff(filename, source, data,
                         resolution=resolution, resolution_unit=resolution_unit,
                         tile_size=tile_size, compression=compression,
                         xml_metadata=xml_metadata, pyramid_levels=4)

        logging.info(f'Image saved as {filename}')

    def _write_tiff(self, filename, source, data,
                  resolution=None, resolution_unit=None, tile_size=None, compression=None,
                  xml_metadata=None, pyramid_levels=0, pyramid_scale=2):

        dim_order = source.get_dimension_order()
        x_index = dim_order.index('x')
        y_index = dim_order.index('y')
        size = data.shape[x_index], data.shape[y_index]

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
            max_size += np.prod(size * scale) * base_size
            scale /= pyramid_scale
        bigtiff = (max_size > 2 ** 32)

        with TiffWriter(filename, bigtiff=bigtiff, ome=is_ome) as writer:
            for level in range(pyramid_levels + 1):
                if level == 0:
                    scale = 1
                    subifds = pyramid_levels
                    subfiletype = None
                else:
                    scale /= pyramid_scale
                    data = resize(data, size * scale)
                    subifds = None
                    subfiletype = 1
                    xml_metadata_bytes = None
                writer.write(data, subifds=subifds, subfiletype=subfiletype,
                             resolution=resolution, resolutionunit=resolution_unit, tile=tile_size,
                             compression=compression, description=xml_metadata_bytes)
