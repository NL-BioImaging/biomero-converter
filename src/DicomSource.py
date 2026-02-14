from datetime import datetime
import numpy as np
import os.path
import pydicom.config
pydicom.config.convert_wrong_length_to_UN = True
from pydicom import dcmread
from pydicom.pixels import pixel_array

from src.ImageSource import ImageSource
from src.util import get_filetitle, redimension_data


class DicomSource(ImageSource):
    """
    ImageSource subclass for reading DICOM files using pydicom.
    """

    def __init__(self, uri, metadata={}):
        super().__init__(uri, metadata)
        if os.path.isfile(uri):
            self.filenames = [uri]
        else:
            self.filenames = [os.path.join(uri, filename) for filename in sorted(os.listdir(uri))]
            uri = self.filenames[0]
        self.dicom = dcmread(uri)

    def walk_dicom(self):
        def callback(dataset, data_element):
            print(f'{data_element.name}: {data_element.value}')

        self.dicom.walk(callback)

    def walk_fileset(self):
        fileset = FileSet(self.dicom)
        for fileinstance in fileset:
            ds = fileinstance.load()
            print(ds.get('filename'), ds.get('StudyDescription'), ds.get('PixelSpacing'))

    def walk_series(self):
        series = self.fileset.find_values("SeriesInstanceUID")
        for serie_id in series:
            fileinstances = self.fileset.find(SeriesInstanceUID=serie_id)
            path = os.path.dirname(fileinstances[0].path)
            data = pixel_array(path)

    def init_metadata(self):
        metadata = {elem.keyword: elem.value for elem in self.dicom.iterall() if elem.keyword}

        self.metadata = metadata
        pixel_array = self.dicom.pixel_array
        shape = list(pixel_array.shape)
        self.is_rgb_type = (metadata.get('PhotometricInterpretation').lower() == 'rgb')
        dim_order = 'yx'
        nchannels = 1
        if self.is_rgb_type:
            if shape[-1] < shape[0]:
                nchannels = shape[-1]
                dim_order = dim_order + 'c'
            else:
                nchannels = shape[0]
                dim_order = 'c' + dim_order
        self.dtype = pixel_array.dtype
        self.pixel_size = {dim:value for dim, value in zip('xy', metadata.get('PixelSpacing', (1, 1)))}
        nz = len(self.filenames)
        if nz > 1:
            dim_order = 'z' + dim_order
            shape = [nz] + shape
            self.pixel_size['z'] = metadata.get('SliceThickness', 1)
        self.shape = shape
        self.nchannels = nchannels
        self.dim_order = dim_order
        self.shapes = [self.shape]
        self.scales = [1]
        if 'ImagePositionPatient' in metadata:
            self.position = {dim: size for dim, size in zip(self.dim_order, metadata['ImagePositionPatient'])}
        else:
            self.position = None
        date_time = metadata.get('AcquisitionDate', '') + metadata.get('AcquisitionTime', '')
        if not date_time:
            date_time = metadata.get('SeriesDate', '') + metadata.get('SeriesTime', '')
        if not date_time:
            date_time = metadata.get('StudyDate', '') + metadata.get('StudyTime', '')
        self.acquisition_datetime = datetime.strptime(date_time, '%Y%m%d%H%M%S')
        self.bits_per_pixel = self.metadata.get('BitsStored', self.dtype.itemsize * 8)

        name = self.metadata.get('SeriesDescription')
        if not name:
            name = self.metadata.get('StudyDescription')
        if not name:
            name = get_filetitle(self.uri)
        self.name = name

        return self.metadata

    def is_screen(self):
        # DICOM files are not multi-well screens
        return False

    def is_rgb(self):
        return self.is_rgb_type

    def get_name(self):
        return self.name

    def get_shape(self):
        return self.shape

    def get_shapes(self):
        return self.shapes

    def get_dtype(self):
        return self.dtype

    def get_scales(self):
        return self.scales

    def get_dim_order(self):
        return self.dim_order

    def get_channels(self):
        if self.is_rgb():
            labels = ['Red', 'Green', 'Blue']
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            return [{'label': label, 'color': color} for label, color in zip(labels, colors)]
        else:
            return [{'label': f'Channel {index}', 'color': [1, 1, 1, 1]} for index in range(self.nchannels)]

    def get_nchannels(self):
        return self.nchannels

    def get_pixel_size_um(self):
        return {dim: size * 1e3 for dim, size in self.pixel_size.items()}

    def get_position_um(self, well_id=None):
        if self.position:
            return {dim: size * 1e3 for dim, size in self.position.items()}
        else:
            return None

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        # https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html#sphx-glr-auto-examples-image-processing-reslice-py
        if 'z' in self.dim_order:
            data = np.zeros(self.shape)
            for index, filename in enumerate(self.filenames):
                data[index] = dcmread(filename).pixel_array
        else:
            data = self.dicom.pixel_array
        return redimension_data(data, self.dim_order, dim_order)
