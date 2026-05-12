import glob
import json
import logging
import os
import pytest
import sys
import tempfile

sys.path.append(os.getcwd())

from converter import init_logging, _convert
from src.helper import create_source
from src.Timer import Timer
from src.util import print_dict, print_hbytes


class TestConvert:
    filenames = ['DB/CellsSmall/experiment.db',
                 'isyntax/small.isyntax',
                 '3DHistech/sample4.mrxs',
                 'EM04573_01small.ome.tif',
                 'incucyte/251125_morphology_XW_2/20251125_morphology_XW2.icarch']
    #filenames = ['incucyte/251125_morphology_XW_2/20251125_morphology_XW2.icarch']
    #filenames = ['incucyte/251125_morphology_XW_2_plate1450.ome.zarr']
    #filenames = ['DB/20220714_TKI_482_small/experiment.db']
    #filenames = ['DB/20220714_TKI_482/experiment.db']
    #filenames = ['XRays/DICOMDIR']
    #filenames = ['XRays/DICOM/PA000001/ST000001/SE000006']
    #filenames = ['XRays/DICOM/PA000002/ST000002/SE000007']
    #filenames = ['rgb.tiff']
    #filenames = ['cideconvolve/Vesicles.ome.tiff']

    input_filenames = ['D:/slides/' + filename for filename in filenames]
    #input_filenames = glob.glob('D:/slides/XRays/DICOM/PA*/ST*/SE*')

    output_formats = ['omezarr3', 'omezarr2', 'ometiff']
    #output_formats = ['omezarr3']

    @pytest.mark.parametrize(
        "input_filename", input_filenames
    )
    def test_source(self, input_filename, verbose=False, **kwargs):
        init_logging('log/biomero_converter.log', verbose=True)
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
        print(input_filename, 'ok')

    @pytest.mark.parametrize(
        "input_filename", input_filenames,
        "output_format", output_formats,
    )
    def test_convert(self, tmp_path, input_filename, output_format, alt_output_folder=None, show_progess=False, verbose=False, **kwargs):
        init_logging('log/biomero_converter.log', verbose=True)
        with Timer(f'convert {input_filename} to {output_format}'):
            output = _convert(input_filename, tmp_path, alt_output_folder=alt_output_folder, output_format=output_format,
                             show_progress=show_progess, verbose=verbose, **kwargs)

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

        output_path = json.loads(output)[0]['full_path']
        target = create_source(output_path)
        metadata = target.init_metadata()
        pixel_size = target.get_pixel_size_um()

        if verbose:
            print('CONVERTED METADATA')
            print(print_dict(metadata))

        source_pixel_size = source.get_pixel_size_um()
        if verbose:
            print(f'Source    pixel size: {source_pixel_size}')
            print(f'Converted pixel size: {pixel_size}')
        assert pixel_size.get('x') == source_pixel_size.get('x')
        assert pixel_size.get('y') == source_pixel_size.get('y')
        if source.is_screen():
            source_wells = kwargs.get('wells', source.get_wells())
            wells = target.get_wells()
            assert list(wells) == list(source_wells)


if __name__ == '__main__':
    # Emulate pytest / fixtures
    from pathlib import Path

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for module in ['ome_zarr', 'zarr', 'numcodecs', 'asyncio']:
        logging.getLogger(module).setLevel(logging.WARNING)

    test = TestConvert()
    for filename in test.input_filenames:
        for output_format in test.output_formats:
            try:
                #test.test_source(filename)
                test.test_convert(Path(tempfile.TemporaryDirectory().name), filename, output_format, show_progess=True)
            except Exception as e:
                print(f'Error converting {filename} to {output_format}: {e}')
                logging.exception(e)
