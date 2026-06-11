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
from src.util import print_dict, print_hbytes


class TestRocrate:
    input_filenames = ['C:/Project/slides/DB/CellsSmall/experiment.db']

    output_formats = ['omezarr3']

    @pytest.mark.parametrize(
        "input_filename", input_filenames,
        "output_format", output_formats,
    )
    def test_convert(self, tmp_path, input_filename, output_format):
        init_logging('log/biomero_converter.log', verbose=True)
        output = _convert(input_filename, tmp_path, output_format=output_format)

        output_path = os.path.join(json.loads(output)[0]['full_path'], 'ro-crate-metadata.json')
        print(open(output_path, encoding='utf-8').read())


if __name__ == '__main__':
    # Emulate pytest / fixtures
    from pathlib import Path

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for module in ['ome_zarr', 'zarr', 'numcodecs', 'asyncio']:
        logging.getLogger(module).setLevel(logging.WARNING)

    test = TestRocrate()
    for input_filename in test.input_filenames:
        for output_format in test.output_formats:
            test.test_convert(Path(tempfile.TemporaryDirectory().name), input_filename, output_format)
