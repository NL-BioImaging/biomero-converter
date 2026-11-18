# Uses https://github.com/anibali/pyisyntax
# which is based on https://github.com/amspath/libisyntax

from datetime import datetime
from isyntax import ISyntax
from xml.etree import ElementTree

from biomero_converter.TiledSource import TiledSource
from biomero_converter.parameters import *
from biomero_converter.util import get_filetitle, xml_content_to_dict, get_bits_type


class ISyntaxSource(TiledSource):
    """
    Loads image and metadata from ISyntax format files.
    """

    # libisyntax reads only on the thread that first opened a file in the process (else: segmentation fault)
    process_reads = True

    def init_metadata(self):
        # read XML metadata header
        data = b''
        block_size = 1024 * 1024
        end_char = b'\x04'   # EOT character
        with open(self.uri, mode='rb') as file:
            done = False
            while not done:
                data_block = file.read(block_size)
                if end_char in data_block:
                    index = data_block.index(end_char)
                    data_block = data_block[:index]
                    done = True
                data += data_block

        self.metadata = xml_content_to_dict(ElementTree.XML(data.decode()))
        if 'DPUfsImport' in self.metadata:
            self.metadata = self.metadata['DPUfsImport']

        image = None
        image_type = ''
        for image0 in self.metadata.get('PIM_DP_SCANNED_IMAGES', []):
            image = image0.get('DPScannedImage', {})
            image_type = image.get('PIM_DP_IMAGE_TYPE').lower()
            if image_type in ['wsi']:
                break

        if image is not None:
            self.image_type = image_type
            nbits = image.get('UFS_IMAGE_BLOCK_HEADER_TEMPLATES', [{}])[0].get('UFSImageBlockHeaderTemplate', {}).get('DICOM_BITS_STORED', 16)
        else:
            self.image_type = ''
            nbits = 16

        self.is_plate = 'screen' in self.image_type or 'plate' in self.image_type or 'wells' in self.image_type

        self.isyntax = self.open_reader(self.uri)
        self.dimensions = self.isyntax.level_dimensions
        self.widths = [width for width, height in self.isyntax.level_dimensions]
        self.heights = [height for width, height in self.isyntax.level_dimensions]
        self.scales = [1 / downsample for downsample in self.isyntax.level_downsamples]

        # original color channels get converted in pyisyntax package to 8-bit RGBA; convert to RGB
        nbits = 8
        self.nchannels = 3
        self.shapes = [(height, width, self.nchannels) for (width, height) in self.dimensions]
        self.shape = self.shapes[0]
        self.dim_order = 'yxc'
        self.is_rgb_channels = True
        self.dtype = get_bits_type(nbits)
        self.pixel_size = {'x': self.isyntax.mpp_x, 'y': self.isyntax.mpp_y}
        self.bits_per_pixel = nbits

        self.name = get_filetitle(self.uri)
        self.acquisition_datetime = datetime.strptime(str(self.metadata.get('DICOM_ACQUISITION_DATETIME')),'%Y%m%d%H%M%S.%f')
        return self.metadata

    def is_screen(self):
        return self.is_plate

    @staticmethod
    def open_reader(uri):
        return ISyntax.open(uri)

    @staticmethod
    def read_rgba_from(reader, x, y, width, height, level=0):
        return reader.read_region(x, y, width, height, level)

    def read_rgba(self, x, y, width, height, level=0):
        return self.read_rgba_from(self.isyntax, x, y, width, height, level)

    def get_position_um(self, well_id=None):
        return {'x': self.isyntax.offset_x, 'y': self.isyntax.offset_y}

    def close(self):
        self.isyntax.close()
        super().close()
