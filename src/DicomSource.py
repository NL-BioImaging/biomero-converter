from datetime import datetime, time
import numpy as np
import os.path
import pydicom.config
pydicom.config.convert_wrong_length_to_UN = True
from pydicom import dcmread
from pydicom.datadict import keyword_for_tag
from pydicom.multival import MultiValue
from pydicom.valuerep import DA, TM

from src.ImageSource import ImageSource
from src.util import get_filetitle, redimension_data, dicom_to_dict


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

    def init_metadata(self):
        metadata = dicom_to_dict(self.dicom)
        self.metadata = metadata
        pixel_array = self.dicom.pixel_array
        shape = list(pixel_array.shape)
        self.is_rgb_type = (metadata.get('PhotometricInterpretation').lower() == 'rgb')
        dim_order = 'yx'
        nchannels = 1
        if self.is_rgb_type:
            # pydicom gives the samples last
            nchannels = shape[-1]
            dim_order = dim_order + 'c'
        self.dtype = pixel_array.dtype
        self.pixel_size = {dim:value for dim, value in zip('xy', metadata.get('PixelSpacing', (1, 1)))}
        if int(metadata.get('NumberOfFrames') or 1) > 1:
            # multi-frame: frames are time points if they increment by frame time (e.g. cine, ultrasound), else slices
            dim_order = ('t' if self._frames_are_time() else 'z') + dim_order
        nfiles = len(self.filenames)
        if nfiles > 1:
            # files are slices
            if 'z' in dim_order:
                shape[0] *= nfiles
            else:
                z_index = 1 if 't' in dim_order else 0
                dim_order = dim_order[:z_index] + 'z' + dim_order[z_index:]
                shape.insert(z_index, nfiles)
        if 'z' in dim_order:
            self.pixel_size['z'] = metadata.get('SliceThickness', 1)
        self.shape = shape
        self.nchannels = nchannels
        self.dim_order = dim_order
        self.shapes = [self.shape]
        self.scales = [1]
        if 'ImagePositionPatient' in metadata:
            self.position = {dim: size for dim, size in zip('xyz', metadata['ImagePositionPatient'])}
        else:
            self.position = None
        self.acquisition_datetime = None
        for prefix in ['Acquisition', 'Series', 'Study']:
            date = metadata.get(prefix + 'Date')
            if date:
                # DICOM DA (YYYYMMDD) and TM (HH[MM[SS[.FFFFFF]]]) values
                time_value = metadata.get(prefix + 'Time')
                self.acquisition_datetime = datetime.combine(DA(date), TM(time_value) if time_value else time())
                break
        self.bits_per_pixel = self.metadata.get('BitsStored', self.dtype.itemsize * 8)

        name = self.metadata.get('SeriesDescription')
        if not name:
            name = self.metadata.get('StudyDescription')
        if not name:
            name = get_filetitle(self.uri)
        self.name = name

        return self.metadata

    def _frames_are_time(self):
        frame_increment_pointer = self.dicom.get('FrameIncrementPointer')
        if frame_increment_pointer is None:
            return False
        tags = frame_increment_pointer if isinstance(frame_increment_pointer, MultiValue) else [frame_increment_pointer]
        return any(keyword_for_tag(tag) in ['FrameTime', 'FrameTimeVector'] for tag in tags)

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

    def get_acquisition_datetime(self):
        return self.acquisition_datetime

    def get_significant_bits(self):
        return self.bits_per_pixel

    def get_time_points(self):
        if 't' in self.dim_order:
            return list(range(self.shape[self.dim_order.index('t')]))
        return []

    def get_rows(self):
        return []

    def get_columns(self):
        return []

    def get_wells(self):
        return []

    def get_fields(self):
        return []

    def get_acquisitions(self):
        return []

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
        # https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html#sphx-glr-auto-examples-image-processing-reslice-py
        if len(self.filenames) > 1:
            arrays = [dcmread(filename).pixel_array for filename in self.filenames]
            if int(self.metadata.get('NumberOfFrames') or 1) > 1 and self.dim_order[0] == 'z':
                data = np.concatenate(arrays)
            else:
                data = np.stack(arrays, axis=self.dim_order.index('z'))
        else:
            data = self.dicom.pixel_array
        return redimension_data(data, self.dim_order, dim_order)
