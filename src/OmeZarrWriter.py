# https://ome-zarr.readthedocs.io/en/stable/python.html#writing-hcs-datasets-to-ome-ngff


#import ngff_zarr
#from ngff_zarr import Multiscales
#from ngff_zarr.v05.zarr_metadata import Axis, Transform, Metadata
#from ome_zarr_models.v05 import Image
#from ome_zarr_models.v05.multiscales import Dataset

import dask.array as da
from ome_zarr import dask_utils
#from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata, write_multiscale
from skimage.transform import resize
import zarr

from src.OmeWriter import OmeWriter
from src.ome_zarr_util import *
from src.parameters import *
from src.util import split_well_name, print_hbytes, get_level_from_scale
from src.WindowScanner import WindowScanner


class OmeZarrWriter(OmeWriter):
    """
    Writer for exporting image or screen data to OME-Zarr format.
    Supports both single images and high-content screening (HCS) plates.
    """

    def __init__(self, zarr_version=2, ome_version='0.4', verbose=False):
        """
        Initialize the OmeZarrWriter.

        Args:
            zarr_version (int): Zarr format version (2 or 3).
            ome_version (str): OME-Zarr metadata version ('0.4' or '0.5').
            verbose (bool): If True, print additional information.
        """
        super().__init__()
        self.zarr_version = zarr_version
        self.ome_version = ome_version
        if ome_version == '0.4':
            from ome_zarr.format import FormatV04
            self.ome_format = FormatV04()
        elif ome_version == '0.5':
            from ome_zarr.format import FormatV05
            self.ome_format = FormatV05()
        else:
            self.ome_format = None
        self.verbose = verbose
        self.dim_order = 'tczyx'

    def write(self, filepath, source, **kwargs):
        """
        Write the provided source data to an OME-Zarr file.

        Args:
            filepath (str): Output path for the Zarr file.
            source: source reader supporting required interface.
            **kwargs: Additional arguments (e.g. wells selection).

        Returns:
            dict: Containing output_path: str Output file path.
        """
        if source.is_screen():
            zarr_root, total_size = self._write_screen(filepath, source, **kwargs)
        else:
            zarr_root, total_size = self._write_image(filepath, source, **kwargs)

        zarr_root.attrs['_creator'] = {'name': 'nl.biomero.OmeZarrWriter', 'version': VERSION}

        if self.verbose:
            print(f'Total data written: {print_hbytes(total_size)}')

        return {'output_path': filepath}

    def _write_screen(self, filepath, source, **kwargs):
        """
        Write a high-content screening (HCS) plate to OME-Zarr.

        Args:
            filepath (str): Output path for the Zarr file.
            source: source reader supporting required interface.
            **kwargs: Additional arguments (e.g., wells).

        Returns:
            tuple: (zarr_root, total_size) where zarr_root is the root group and total_size is bytes written.
        """
        #zarr_location = parse_url(filename, mode='w', fmt=self.ome_format)
        zarr_location = filepath
        zarr_root = zarr.open_group(zarr_location, mode='w', zarr_version=self.zarr_version)

        row_names = [chr(ord('A') + index) for index
                     in range(max([ord(row_name.upper()) - ord('A') for row_name in source.get_rows()]) + 1)]
        col_names = [str(index) for index
                     in range(1, max([int(col) for col in source.get_columns()]) + 1)]
        wells = kwargs.get('wells', source.get_wells())
        well_paths = ['/'.join(split_well_name(well)) for well in wells]
        fields = list(map(str, source.get_fields()))

        acquisitions = source.get_acquisitions()
        name = source.get_name()
        write_plate_metadata(zarr_root, row_names, col_names, well_paths,
                             name=name, field_count=len(fields), acquisitions=acquisitions,
                             fmt=self.ome_format)
        total_size = 0
        for well_id in wells:
            row, col = split_well_name(well_id)
            row_group = zarr_root.require_group(str(row))
            well_group = row_group.require_group(str(col))
            write_well_metadata(well_group, fields, fmt=self.ome_format)
            position = source.get_position_um(well_id)
            for field_id in fields:
                image_group = well_group.require_group(field_id)
                data = source.get_data(self.dim_order, well_id=well_id, field_id=field_id)
                window_scanner = WindowScanner()
                window = source.get_image_window(window_scanner, well_id=well_id, field_id=field_id, data=data)
                size = self._write_data(image_group, data, source, window, position=position)
                total_size += size

        return zarr_root, total_size

    def _write_image(self, filepath, source, **kwargs):
        """
        Write a single image to OME-Zarr.

        Args:
            filepath (str): Output path for the Zarr file.
            source: source reader for image data.
            **kwargs: Additional arguments.

        Returns:
            tuple: (zarr_root, size) where zarr_root is the root group and size is bytes written.
        """
        #zarr_location = parse_url(filename, mode='w', fmt=self.ome_format)
        zarr_location = filepath
        zarr_root = zarr.open_group(zarr_location, mode='w', zarr_version=self.zarr_version)

        pyramid_data = []
        scale = 1
        last_level = None
        for index in range(PYRAMID_LEVELS + 1):
            level, rescale = get_level_from_scale(source.get_scales(), scale)
            if level != last_level:
                data = source.get_data_as_dask(self.dim_order, level=level)
                last_level = level
            if index == 0:
                shape0 = data.shape
            if rescale != 1:
                shape = list(shape0)
                shape[-2:] = np.multiply(shape0[-2:], scale).astype(int)
                if isinstance(data, da.Array):
                    data = dask_utils.resize(data, shape)
                else:
                    data = resize(data, shape, preserve_range=True).astype(data.dtype)
            pyramid_data.append(data)
            scale /= PYRAMID_DOWNSCALE

        window_scanner = WindowScanner()
        window = source.get_image_window(window_scanner)
        size = self._write_data(zarr_root, pyramid_data, source, window, position=source.get_position_um())
        return zarr_root, size

    def _write_data(self, group, data, source, window, position=None):
        """
        Write image data and metadata to a Zarr group.

        Args:
            group: Zarr group to write into.
            data: Image data array.
            source: source reader.
            window: Image window information.
            position: Optional position information.

        Returns:
            int: Number of bytes written.
        """
        dim_order = self.dim_order
        dtype = source.get_dtype()
        channels = source.get_channels()
        nchannels = source.get_nchannels()
        is_rgb = source.is_rgb()

        axes = create_axes_metadata(dim_order)
        pixel_size_scales, scaler = self._create_scale_metadata(source, dim_order, position)
        metadata = {'omero': create_channel_metadata(dtype, channels, nchannels, is_rgb, window, self.ome_version),
                    'method': scaler.method}

        is_pyramid = isinstance(data, list)
        if is_pyramid:
            data0 = data[0]
        else:
            data0 = data
        storage_options = None
        if self.zarr_version >= 3:
            if not hasattr(data0, 'chunksize'):
                chunks = []
                shards = []
                for dim, n in zip(dim_order, data0.shape):
                    if dim in 'xy':
                        chunks += [ZARR_CHUNK_SIZE]
                        shards += [ZARR_CHUNK_SIZE * ZARR_SHARD_MULTIPLIER]
                    else:
                        chunks += [1]
                        shards += [1]
                storage_options = {'chunks': chunks, 'shards': shards}

        size = data0.size * data0.itemsize
        if is_pyramid:
            #images = [Image.fromarray(data1) for data1 in data]
            #ngff_zarr.from_ngff_zarr() # use this to see construction
            #axes1 = [Axis()]
            #datasets1 = [Dataset()]
            #coordinateTransformations1 = Transform()
            #metadata = Metadata(axes1, datasets1, coordinateTransformations1)
            #multiscales = Multiscales(images, metadata)
            #ngff_zarr.to_ngff_zarr(group, multiscales=multiscales)

            write_multiscale(pyramid=data, group=group, axes=axes, coordinate_transformations=pixel_size_scales,
                            fmt=self.ome_format, storage_options=storage_options,
                            name=source.get_name(), metadata=metadata)
        else:
            write_image(image=data, group=group, axes=axes, coordinate_transformations=pixel_size_scales,
                        scaler=scaler, fmt=self.ome_format, storage_options=storage_options,
                        name=source.get_name(), metadata=metadata)
        return size

    def _create_scale_metadata(self, source, dim_order, translation, scaler=None):
        """
        Create coordinate transformation metadata for multiscale images.

        Args:
            source: source reader.
            dim_order (str): Dimension order string.
            translation: Translation or position information.
            scaler: Optional Scaler object.

        Returns:
            tuple: (pixel_size_scales, scaler)
        """
        if scaler is None:
            scaler = Scaler(downscale=PYRAMID_DOWNSCALE, max_layer=PYRAMID_LEVELS)
        pixel_size_scales = []
        factor = 1
        for i in range(scaler.max_layer + 1):
            pixel_size_scales.append(
                create_transformation_metadata(dim_order, source.get_pixel_size_um(),
                                               factor, translation))
            factor *= scaler.downscale
        return pixel_size_scales, scaler
