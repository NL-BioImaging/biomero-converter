# TODO: requires proper implementation including ome xml metadata

import logging
import os.path
from tifffile import tifffile

from src.OmeWriter import OmeWriter
from src.util import *


class OmeTiffWriter(OmeWriter):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def write(self, filename, source, name=None, well_id=None, field_id=None, tiff_compression=None, **kwargs):
        filepath, filename = os.path.split(filename)
        filetitle, ext = os.path.splitext(filename)

        filename = f'{filetitle}'
        filename += f'_{pad_leading_zero(well_id)}'
        if field_id is not None and field_id >= 0:
            filename += f'_{pad_leading_zero(field_id)}'

        filename = os.path.join(filepath, filename + ext)

        data = source.get_data(well_id, field_id)

        if ext.lower() in ['.tif', '.tiff']:
            with tifffile.TiffWriter(filename) as tif:
                tif.write(data, compression=tiff_compression, dtype=source.metadata['dtype'])

        logging.info(f'Image saved as {filename}')
