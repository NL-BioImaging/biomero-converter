import json
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os
import pytest
import sys
import tempfile

sys.path.append(os.getcwd())

from converter import init_logging, convert
from src.helper import create_source
from src.TiffSource import TiffSource
from src.Timer import Timer
from src.util import print_dict, print_hbytes


class TestConvert:
    filenames = ['DB/TestData1/experiment.db', 'isyntax/small.isyntax', 'EM04573_01small.ome.tif']
    filenames = ['3DHistech/sample4.mrxs']
    input_filenames = ['E:/slides/' + filename for filename in filenames]

    output_formats = ['omezarr2', 'omezarr3', 'ometiff']

    @pytest.mark.parametrize(
        "input_filename", input_filenames,
        "output_format", output_formats,
    )
    def test_convert(self, tmp_path, input_filename, output_format, alt_output_folder=None, show_progess=False, verbose=False, **kwargs):
        init_logging('log/db_to_zarr.log', verbose=True)
        with Timer(f'convert {input_filename} to {output_format}'):
            output = convert(input_filename, tmp_path, alt_output_folder=alt_output_folder, output_format=output_format,
                             show_progress=show_progess, verbose=verbose, max_attempts=1, **kwargs)

        source = create_source(input_filename)
        metadata = source.init_metadata()
        if verbose:
            print('SOURCE METADATA')
            print(print_dict(metadata))
            print()
            if source.is_screen():
                print(source.print_well_matrix())
                print(source.print_timepoint_well_matrix())
            print(f'Total data size:    {print_hbytes(source.get_total_data_size())}')

        #print(print_dict(metadata))

        output_path = json.loads(output)[0]['full_path']
        if 'tif' in output_format:
            reader = TiffSource(output_path)
            metadata = reader.init_metadata()
            pixel_size = reader.get_pixel_size_um()
            if source.is_screen():
                wells = reader.get_wells()
        else:
            reader = Reader(parse_url(output_path))
            node = list(reader())[0]
            metadata = node.metadata
            axes = [axis['name'] for axis in metadata['axes']]
            pixel_sizes0 = [transform for transform in metadata['coordinateTransformations'][0] if transform['type'] == 'scale'][0]['scale']
            pixel_size = {axis: pixel_size for axis, pixel_size in zip(axes, pixel_sizes0) if axis in 'xyz'}
            if source.is_screen():
                wells = [well['path'].replace('/', '') for well in metadata['metadata']['plate']['wells']]

        if verbose:
            print('CONVERTED METADATA')
            print(print_dict(metadata))

        if '2' in output_format:
            assert float(reader.zarr.version) == 0.4
        elif '3' in output_format:
            assert float(reader.zarr.version) >= 0.5

        if '2' in output_format:
            assert float(node.zarr.version) == 0.4
        elif '3' in output_format:
            assert float(node.zarr.version) >= 0.5

        source_pixel_size = source.get_pixel_size_um()
        source_wells = kwargs.get('wells', source.get_wells())
        if verbose:
            print(f'Source    pixel size: {source_pixel_size}')
            print(f'Converted pixel size: {pixel_size}')
        assert pixel_size.get('x') == source_pixel_size.get('x')
        assert pixel_size.get('y') == source_pixel_size.get('y')
        if source.is_screen():
            assert list(wells) == list(source_wells)


if __name__ == '__main__':
    # Emulate pytest / fixtures
    from pathlib import Path

    test = TestConvert()
    for filename in test.input_filenames:
        for output_format in test.output_formats:
            test.test_convert(Path(tempfile.TemporaryDirectory().name), filename, output_format, show_progess=True)
