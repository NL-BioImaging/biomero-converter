import json
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import pytest
import tempfile

from converter import init_logging, convert
from src.helper import create_source
from src.TiffSource import TiffSource
from src.Timer import Timer
from src.util import print_dict, print_hbytes


class TestConvert:
    #basedir = 'C:/Project/slides/DB/'
    basedir = 'D:/slides/DB/'
    #basedir = 'C:/Project/slides/Ome-tiff/'
    #basedir = 'E:/Personal/Crick/slides/test_images/'
    #basedir = 'D:/slides/isyntax/'

    filename = 'TestData1/experiment.db'
    #filename = '2ChannelPlusTL/experiment.db'
    #filename = 'PicoData16ProcCoverag/experiment.db'
    #filename = '241209 - TC1 TC9 test MSP MUB/experiment.db'
    #filename = '20220714_TKI_482/experiment.db'
    #filename = 'Cells/experiment.db'
    #filename = 'NIRHTa-001.ome.tiff'
    #filename = 'signed single-channel.ome.tiff'
    #filename = 'volumetric Broken_NE_cropped.tif'
    #filename = 'small.isyntax'
    #filename = 'test-isyntax.isyntax'

    input_filename = basedir + filename

    @pytest.mark.parametrize(
        "input_filename, output_format",
        [
            (
                input_filename,
                'omezarr2',
            ),
            (
                input_filename,
                'omezarr3',
            ),
        ],
    )
    def test_convert(self, tmp_path, input_filename, output_format, alt_output_folder=None, show_progess=True, verbose=False):
        init_logging('log/db_to_zarr.log', verbose=verbose)
        with Timer(f'convert {input_filename} to {output_format}'):
            output = convert(input_filename, tmp_path, alt_output_folder=alt_output_folder, output_format=output_format, show_progress=show_progess, verbose=verbose)

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
        source_pixel_size = source.get_pixel_size_um()
        if source.is_screen():
            source_wells = source.get_wells()

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

        print(f'Source    pixel size: {source_pixel_size}')
        print(f'Converted pixel size: {pixel_size}')
        assert pixel_size.get('x') == source_pixel_size.get('x')
        assert pixel_size.get('y') == source_pixel_size.get('y')
        if source.is_screen():
            assert wells == source_wells


if __name__ == '__main__':
    # Emulate pytest / fixtures
    from pathlib import Path

    test = TestConvert()
    input_filename = test.input_filename
    test.test_convert(Path(tempfile.TemporaryDirectory().name), input_filename, 'ometiff', alt_output_folder=tempfile.TemporaryDirectory().name)
    test.test_convert(Path(tempfile.TemporaryDirectory().name), input_filename, 'omezarr2', alt_output_folder=tempfile.TemporaryDirectory().name)
    test.test_convert(Path(tempfile.TemporaryDirectory().name), input_filename, 'omezarr3', alt_output_folder=tempfile.TemporaryDirectory().name)
