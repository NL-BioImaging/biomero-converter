from datetime import datetime, time
import logging
import numpy as np
import os.path
import pydicom.config
pydicom.config.convert_wrong_length_to_UN = True
from pydicom import dcmread
from pydicom.datadict import keyword_for_tag
from pydicom.encaps import generate_frames
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.valuerep import DA, TM

from src.ImageSource import ImageSource
from src.util import get_filetitle, redimension_data, dicom_to_dict


# compressed transfer syntaxes imagecodecs can decode, for when pydicom has no decoder plugin installed
IMAGECODECS_DECODERS = {
    '1.2.840.10008.1.2.4.57': 'ljpeg_decode',   # JPEG Lossless
    '1.2.840.10008.1.2.4.70': 'ljpeg_decode',   # JPEG Lossless, first-order prediction
    '1.2.840.10008.1.2.4.80': 'jpegls_decode',  # JPEG-LS lossless
    '1.2.840.10008.1.2.4.81': 'jpegls_decode',  # JPEG-LS near-lossless
    '1.2.840.10008.1.2.4.90': 'jpeg2k_decode',  # JPEG 2000 lossless
    '1.2.840.10008.1.2.4.91': 'jpeg2k_decode',  # JPEG 2000
}


class DicomSource(ImageSource):
    """
    ImageSource subclass for reading DICOM files using pydicom.

    A folder (or the folder of a DICOMDIR) can hold several series: each series is an image, selected by its
    SeriesInstanceUID (image_uuid), its files the slices in order of position.
    """

    def __init__(self, uri, metadata={}, image_uuid=None, **kwargs):
        """
        Initialize DicomSource.

        Args:
            uri (str): Path to a DICOM file, a folder of DICOM files or a DICOMDIR.
            metadata (dict): Optional metadata dictionary.
            image_uuid (str, optional): SeriesInstanceUID of the series to read (default: the first).
        """
        super().__init__(uri, metadata)
        series = self._find_series(uri)
        if not series:
            raise ValueError(f'No DICOM images found in {uri}')
        if image_uuid:
            if image_uuid not in series:
                raise ValueError(f'Series UID {image_uuid} not found in {uri}')
            self.series_uid = image_uuid
        else:
            self.series_uid = next(iter(series))
        self.nseries = len(series)
        self.filenames, nleft_out = series[self.series_uid]
        if nleft_out:
            logging.warning(f'DICOM series {self.series_uid}: using {len(self.filenames)} slices of one orientation, '
                            f'leaving out {nleft_out} (e.g. a reference image)')
        self.dicom = dcmread(self.filenames[0])

    @staticmethod
    def get_series(uri):
        """
        Find the series in a DICOM file, folder or DICOMDIR.

        Returns:
            dict: Per SeriesInstanceUID, its filenames in order of slice position.
        """
        return {series_uid: filenames for series_uid, (filenames, _) in DicomSource._find_series(uri).items()}

    @staticmethod
    def _find_series(uri):
        # per SeriesInstanceUID: its filenames in order of slice position, and the number of files left out
        if os.path.isfile(uri) and os.path.basename(uri).upper() != 'DICOMDIR':
            dicom = dcmread(uri, stop_before_pixels=True)
            return {str(dicom.get('SeriesInstanceUID', '')): ([uri], 0)}
        folder = uri if os.path.isdir(uri) else os.path.dirname(uri)
        groups = {}
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                path = os.path.join(root, filename)
                try:
                    dicom = dcmread(path, stop_before_pixels=True)
                except (InvalidDicomError, OSError):
                    continue
                if 'Rows' not in dicom:
                    # no image (e.g. DICOMDIR, report)
                    continue
                series_uid = str(dicom.get('SeriesInstanceUID', path))
                orientation = tuple(round(float(value), 3) for value in dicom.get('ImageOrientationPatient', []))
                groups.setdefault(series_uid, {}).setdefault((orientation, dicom.Rows, dicom.Columns), []).append(
                    (path, dicom))
        series = []
        for series_uid, series_groups in groups.items():
            # slices of one orientation and size: the largest group (e.g. not a reformat's reference image)
            group = max(series_groups.values(), key=len)
            nleft_out = sum(map(len, series_groups.values())) - len(group)
            group.sort(key=lambda item: DicomSource._get_slice_order(item[1]))
            first = group[0][1]
            order = (str(first.get('StudyInstanceUID', '')), int(first.get('SeriesNumber') or 0), group[0][0])
            series.append((order, series_uid, [path for path, _ in group], nleft_out))
        return {series_uid: (filenames, nleft_out) for _, series_uid, filenames, nleft_out in sorted(series)}

    @staticmethod
    def _get_slice_position(dicom):
        # position along the slice normal (the cross product of the image row and column directions)
        orientation = dicom.get('ImageOrientationPatient')
        position = dicom.get('ImagePositionPatient')
        if orientation is None or position is None or len(orientation) != 6:
            return None
        normal = np.cross(np.array(orientation[:3], dtype=float), np.array(orientation[3:], dtype=float))
        return float(np.dot(normal, np.array(position, dtype=float)))

    @staticmethod
    def _get_slice_order(dicom):
        position = DicomSource._get_slice_position(dicom)
        return (position if position is not None else 0, int(dicom.get('InstanceNumber') or 0))

    def init_metadata(self):
        metadata = dicom_to_dict(self.dicom)
        self.metadata = metadata
        pixel_array = self._read_pixels(self.dicom)
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
            self.pixel_size['z'] = self._get_slice_spacing()
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
            # the folder's name for a DICOMDIR
            is_dicomdir = os.path.basename(self.uri).upper() == 'DICOMDIR'
            name = get_filetitle(os.path.dirname(self.uri) if is_dicomdir else self.uri)
        if self.nseries > 1:
            name = f"{name}_{self.metadata.get('SeriesNumber', self.series_uid)}"
        self.name = name

        return self.metadata

    def _get_slice_spacing(self):
        # slice spacing from the slice positions, else as stated
        if len(self.filenames) > 1:
            positions = [self._get_slice_position(dcmread(filename, stop_before_pixels=True))
                         for filename in self.filenames]
            if None not in positions:
                steps = [step for step in np.abs(np.diff(positions)) if step > 0]
                if steps:
                    return float(np.median(steps))
        spacing = self.metadata.get('SpacingBetweenSlices')
        if spacing:
            return abs(spacing)
        return self.metadata.get('SliceThickness', 1)

    def _frames_are_time(self):
        frame_increment_pointer = self.dicom.get('FrameIncrementPointer')
        if frame_increment_pointer is None:
            return False
        tags = frame_increment_pointer if isinstance(frame_increment_pointer, MultiValue) else [frame_increment_pointer]
        return any(keyword_for_tag(tag) in ['FrameTime', 'FrameTimeVector'] for tag in tags)

    @staticmethod
    def _read_pixels(dicom):
        try:
            return dicom.pixel_array
        except RuntimeError:
            # no pydicom decoder plugin for this compression: decode with imagecodecs instead
            transfer_syntax = str(dicom.file_meta.TransferSyntaxUID)
            if transfer_syntax not in IMAGECODECS_DECODERS:
                raise
            import imagecodecs
            decode = getattr(imagecodecs, IMAGECODECS_DECODERS[transfer_syntax])
            nframes = int(dicom.get('NumberOfFrames') or 1)
            frames = [decode(frame) for frame in generate_frames(dicom.PixelData, number_of_frames=nframes)]
            data = np.stack(frames) if nframes > 1 else frames[0]
            if dicom.get('PixelRepresentation') == 1 and data.dtype.kind == 'u':
                # signed values: sign-extend the stored bits
                data = data.view(data.dtype.str.replace('u', 'i'))
                unused_bits = data.dtype.itemsize * 8 - int(dicom.get('BitsStored') or data.dtype.itemsize * 8)
                if unused_bits > 0:
                    data = (data << unused_bits) >> unused_bits
            return data

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

    def get_acquisition_metadata(self):
        # all DICOM attributes as they are (patient details included), for mapping onto the imaging metadata model
        return self.metadata

    def get_source_metadata(self):
        # the attributes but the padding that fills out the file (zeros), meaningless once the pixels are converted
        return {key: value for key, value in self.metadata.items() if key != 'DataSetTrailingPadding'}

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
            arrays = [self._read_pixels(dcmread(filename)) for filename in self.filenames]
            if int(self.metadata.get('NumberOfFrames') or 1) > 1 and self.dim_order[0] == 'z':
                data = np.concatenate(arrays)
            else:
                data = np.stack(arrays, axis=self.dim_order.index('z'))
        else:
            data = self._read_pixels(self.dicom)
        return redimension_data(data, self.dim_order, dim_order)
