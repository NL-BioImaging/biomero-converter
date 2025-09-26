# https://ome-zarr.readthedocs.io/en/stable/python.html#writing-hcs-datasets-to-ome-ngff

#from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
import zarr

from src.OmeWriter import OmeWriter
from src.ome_zarr_util import *
from src.parameters import *
from src.util import split_well_name, print_hbytes
from src.WindowScanner import WindowScanner


class OmeZarrWriter(OmeWriter):
    def __init__(self, zarr_version=2, ome_version='0.4', verbose=False):
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

    def write(self, filepath, source, **kwargs):
        if source.is_screen():
            zarr_root, total_size = self._write_screen(filepath, source, **kwargs)
        else:
            zarr_root, total_size = self._write_image(filepath, source)

        zarr_root.attrs['_creator'] = {'name': 'nl.biomero.OmeZarrWriter', 'version': VERSION}

        if self.verbose:
            print(f'Total data written: {print_hbytes(total_size)}')

        return filepath

    def _write_screen(self, filepath, source, **kwargs):
        #zarr_location = parse_url(filename, mode='w', fmt=self.ome_format)
        zarr_location = filepath
        zarr_root = zarr.open_group(zarr_location, mode='w', zarr_version=self.zarr_version)

        row_names = source.get_rows()
        col_names = source.get_columns()
        wells = source.get_wells()
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
            for field in fields:
                image_group = well_group.require_group(field)
                data = source.get_data(well_id, field)
                size = self._write_data(image_group, data, source, position)
                total_size += size

        return zarr_root, total_size

    def _write_image(self, filepath, source):
        #zarr_location = parse_url(filename, mode='w', fmt=self.ome_format)
        zarr_location = filepath
        zarr_root = zarr.open_group(zarr_location, mode='w', zarr_version=self.zarr_version)

        data = source.get_data(as_dask=True)
        size = self._write_data(zarr_root, data, source)
        return zarr_root, size

    def _write_data(self, group, data, source, position=None):
        window_scanner = WindowScanner()
        dim_order = source.get_dim_order()
        dtype = source.get_dtype()
        channels = source.get_channels()
        nchannels = source.get_nchannels()
        is_rgb = source.is_rgb()

        axes = create_axes_metadata(dim_order)
        pixel_size_scales, scaler = self._create_scale_metadata(source, dim_order, position)
        window_scanner.process(data, dim_order)
        window = window_scanner.get_window()
        metadata = {'omero': create_channel_metadata(dtype, channels, nchannels, is_rgb, window, self.ome_version),
                    'metadata': {'method': scaler.method}}

        storage_options = None
        if self.zarr_version >= 3:
            if not hasattr(data, 'chunksize'):
                chunks = []
                shards = []
                for dim, n in zip(dim_order, data.shape):
                    if dim in 'xy':
                        chunks += [ZARR_CHUNK_SIZE]
                        shards += [ZARR_CHUNK_SIZE * ZARR_SHARD_MULTIPLIER]
                    else:
                        chunks += [1]
                        shards += [1]
                storage_options = {'chunks': chunks, 'shards': shards}

        size = data.size * data.itemsize
        write_image(image=data, group=group, axes=axes, coordinate_transformations=pixel_size_scales,
                    scaler=scaler, fmt=self.ome_format, storage_options=storage_options,
                    name=source.get_name(), metadata=metadata)

        return size

    def _create_scale_metadata(self, source, dim_order, translation, scaler=None):
        if scaler is None:
            scaler = Scaler()
        pixel_size_scales = []
        scale = 1
        for i in range(scaler.max_layer + 1):
            pixel_size_scales.append(
                create_transformation_metadata(dim_order, source.get_pixel_size_um(),
                                               scale, translation))
            scale /= scaler.downscale
        return pixel_size_scales, scaler
