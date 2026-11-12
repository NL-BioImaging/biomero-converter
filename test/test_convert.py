import glob
import json
import logging
import numpy as np
import os
import pytest
import sys
import tempfile


from biomero_converter.converter import init_logging, _convert
from biomero_converter.helper import LEICA_EXTENSIONS, create_source, get_incucyte_plates
from biomero_converter.Timer import Timer
from biomero_converter.util import print_dict, print_hbytes


class TestConvert:
    filenames = ['DB/CellsSmall/experiment.db',
                 'isyntax/small.isyntax',
                 '3DHistech/sample4.mrxs',
                 'tiff/EM04573_01small.ome.tif',
                 'incucyte/251125_morphology_XW_2/20251125_morphology_XW2.icarch',
                 'Leica/lif/3dtiny.lif']

    input_filenames = ['C:/Project/slides/' + filename for filename in filenames]

    output_formats = ['omezarr06', 'omezarr05', 'omezarr04', 'ometiff']

    # source parameters per input file (by end of path)
    source_kwargs = {'Leica/lif/TestTileScan16bit.lif': {'position': 0}}   # tile scan with gaps: separate positions

    def get_source_kwargs(self, input_filename):
        return next((kwargs for end, kwargs in self.source_kwargs.items()
                     if input_filename.replace('\\', '/').endswith(end)), {})

    @pytest.mark.parametrize("input_filename", input_filenames)
    def test_source(self, input_filename, verbose=False, **kwargs):
        init_logging('log/biomero_converter.log', verbose=True)
        source = create_source(input_filename, **self.get_source_kwargs(input_filename))
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

    @pytest.mark.parametrize("input_filename", input_filenames)
    @pytest.mark.parametrize("output_format", output_formats)
    def test_convert(self, tmp_path, input_filename, output_format, alt_output_folder=None, show_progess=False, verbose=False, **kwargs):
        init_logging('log/biomero_converter.log', verbose=True)
        source_kwargs = self.get_source_kwargs(input_filename)
        with Timer(f'convert {input_filename} to {output_format}'):
            output = _convert(input_filename, tmp_path, alt_output_folder=alt_output_folder, output_format=output_format,
                             show_progress=show_progess, verbose=verbose, **source_kwargs, **kwargs)

        # every output (e.g. each Leica image or Incucyte plate) against its source image
        results = json.loads(output)
        output_source_kwargs = self.get_output_source_kwargs(input_filename, source_kwargs, len(results))
        assert len(results) == len(output_source_kwargs)
        for result, output_kwargs in zip(results, output_source_kwargs):
            source = create_source(input_filename, **output_kwargs)
            metadata = source.init_metadata()
            if verbose:
                print('SOURCE METADATA')
                print(print_dict(metadata))
                print()
                if source.is_screen():
                    print(source.print_well_matrix())
                    print(source.print_timepoint_well_matrix())
                print(f'Total data size:    {print_hbytes(source.get_total_data_size())}')

            target = create_source(result['full_path'])
            metadata = target.init_metadata()
            if verbose:
                print('CONVERTED METADATA')
                print(print_dict(metadata))

            self.assert_same_image(source, target, wells=kwargs.get('wells'), verbose=verbose)
            source.close()
            target.close()

    @staticmethod
    def get_output_source_kwargs(input_filename, source_kwargs, noutputs):
        # source parameters of each output, as converter.py converts each Leica image / Incucyte plate
        input_ext = os.path.splitext(input_filename)[1].lower()
        if (input_ext in LEICA_EXTENSIONS and not source_kwargs.get('image_uuid')
                and source_kwargs.get('image_index') is None):
            return [{**source_kwargs, 'image_index': index} for index in range(noutputs)]
        if input_ext == '.icarch' and source_kwargs.get('plate_id') is None:
            return [{**source_kwargs, 'plate_id': plate_id} for plate_id in get_incucyte_plates(input_filename)]
        return [source_kwargs]

    @staticmethod
    def assert_same_image(source, target, wells=None, verbose=False):
        source_pixel_size = source.get_pixel_size_um()
        pixel_size = target.get_pixel_size_um()
        if verbose:
            print(f'Source    pixel size: {source_pixel_size}')
            print(f'Converted pixel size: {pixel_size}')
        assert pixel_size.get('x') == source_pixel_size.get('x')
        assert pixel_size.get('y') == source_pixel_size.get('y')

        if source.is_screen():
            source_wells = wells if wells is not None else source.get_wells()
            assert list(target.get_wells()) == list(source_wells)
            for well_id in source_wells:
                for field_id in source.get_fields():
                    source_data = np.asarray(source.get_data('tczyx', well_id=well_id, field_id=field_id))
                    data = np.asarray(target.get_data('tczyx', well_id=well_id, field_id=field_id))
                    assert data.shape == source_data.shape, f'well {well_id} field {field_id}'
                    assert np.array_equal(data, source_data), f'well {well_id} field {field_id}'
        else:
            source_data = np.asarray(source.get_data('tczyx'))
            data = np.asarray(target.get_data('tczyx'))
            assert data.shape == source_data.shape
            assert np.array_equal(data, source_data)

            source_position = source.get_position_um() or {}
            position = target.get_position_um() or {}
            for dim, value in source_position.items():
                assert position.get(dim) == pytest.approx(value), f'position {dim}'


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


def test_convert_retry_reports_first_error(monkeypatch, caplog):
    # A retry can fail differently as a side effect of the first failure; the first error must not be lost
    from biomero_converter import converter
    errors = iter([FileNotFoundError('schema.yaml'), ImportError('circular import'), ImportError('circular import')])

    def failing_convert(*args, **kwargs):
        raise next(errors)

    monkeypatch.setattr(converter, '_convert', failing_convert)
    with caplog.at_level(logging.ERROR), pytest.raises(Exception) as exc_info:
        converter.convert('input.lof', 'output', max_attempts=3)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    assert 'schema.yaml' in str(exc_info.value)
    attempt_logs = [record for record in caplog.records if 'Conversion attempt' in record.getMessage()]
    assert len(attempt_logs) == 3
    assert attempt_logs[0].exc_info is not None


def test_convert_retry_succeeds(monkeypatch):
    from biomero_converter import converter
    results = iter([RuntimeError('transient'), '[{"name": "ok"}]'])

    def flaky_convert(*args, **kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(converter, '_convert', flaky_convert)
    assert converter.convert('input.tif', 'output', max_attempts=3) == '[{"name": "ok"}]'


def test_default_output_format(tmp_path):
    # Without an explicit format, convert() writes DEFAULT_OUTPUT_FORMAT (OME-Zarr 0.5)
    import tifffile
    from biomero_converter.converter import convert
    from biomero_converter.parameters import DEFAULT_OUTPUT_FORMAT
    assert DEFAULT_OUTPUT_FORMAT == 'omezarr05'
    input_filename = str(tmp_path / 'input.tif')
    tifffile.imwrite(input_filename, np.zeros((32, 32), dtype=np.uint8))
    result = json.loads(convert(input_filename, str(tmp_path / 'output')))
    with open(os.path.join(result[0]['full_path'], 'zarr.json')) as file:
        zarr_metadata = json.load(file)
    assert zarr_metadata['attributes']['ome']['version'] == '0.5'


def test_cli_default_output_format(monkeypatch):
    # The command line default is the same constant
    from biomero_converter import main
    from biomero_converter.parameters import DEFAULT_OUTPUT_FORMAT
    captured = {}
    monkeypatch.setattr(main, 'convert', lambda *args, **kwargs: captured.update(kwargs) or '[{}]')
    monkeypatch.setattr(main, 'init_logging', lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, 'argv', ['biomero-converter', '--inputfile', 'in.tif', '--outputfolder', 'out'])
    with pytest.raises(SystemExit):
        main.main()
    assert captured['output_format'] == DEFAULT_OUTPUT_FORMAT
