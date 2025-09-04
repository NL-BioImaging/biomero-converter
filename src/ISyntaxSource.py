# Uses https://github.com/anibali/pyisyntax
# which is based on https://github.com/amspath/libisyntax


from isyntax import ISyntax
import numpy as np
from xml.etree import ElementTree

from ImageSource import ImageSource
from util import get_filetitle, xml_content_to_dict


class ISyntaxSource(ImageSource):
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
            nbits = int(np.ceil(nbits / 8)) * 8
        else:
            self.image_type = ''
            nbits = 16

        self.is_plate = 'screen' in self.image_type or 'plate' in self.image_type or 'wells' in self.image_type

        # original color channels get converted in pyisyntax package to 8-bit RGBA
        nbits = 8
        self.dim_order = 'yxc'
        self.channels = []
        self.nchannels = 4

        self.isyntax = ISyntax.open(self.uri)
        self.width, self.height = self.isyntax.dimensions
        self.shape = 1, self.nchannels, 1, self.height, self.width
        self.dtype = np.dtype(f'uint{nbits}')

        return self.metadata

    def is_screen(self):
        return self.is_plate

    def get_shape(self):
        return self.shape

    def get_data(self, well_id=None, field_id=None):
        return self.isyntax.read_region(0, 0, self.width, self.height)

    def get_name(self):
        return get_filetitle(self.uri)

    def get_dim_order(self):
        return self.dim_order

    def get_pixel_size_um(self):
        return {'x': self.isyntax.mpp_x, 'y': self.isyntax.mpp_y}

    def get_dtype(self):
        return self.dtype

    def get_position_um(self, well_id=None):
        return {}

    def get_channels(self):
        return self.channels

    def get_nchannels(self):
        return self.nchannels

    def get_rows(self):
        return []

    def get_columns(self):
        return []

    def get_wells(self):
        return []

    def get_time_points(self):
        return 0

    def get_fields(self):
        return []

    def get_acquisitions(self):
        return []

    def get_total_data_size(self):
        total_size = np.prod(self.shape)
        if self.is_plate:
            total_size *= len(self.get_wells()) * len(self.get_fields())
        return total_size

    def close(self):
        self.isyntax.close()
