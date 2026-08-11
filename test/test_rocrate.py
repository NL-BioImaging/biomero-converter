import glob
import logging
import os
import pytest
from rocrate.model.metadata import BASENAME
import sys
import tempfile

sys.path.append(os.getcwd())

from converter import init_logging
from src.helper import create_source
from src.rocrate_utils import create_ro_crate


class TestRocrate:
    input_filenames = ['C:/Project/slides/DB/CellsSmall/experiment.db']
    #input_filenames = ['C:/Project/slides/tiff/DNAcropSmall.ome.tiff']
    input_filenames = glob.glob('C:/Project/slides/tiff/*.tif*')

    @pytest.mark.parametrize(
        "input_filename", input_filenames
    )
    def test_rocrate(self, tmp_path, input_filename):
        init_logging('log/biomero_converter.log', verbose=True)
        source = create_source(input_filename)
        source.init_metadata()
        create_ro_crate(source=source, dest_path=tmp_path)

        print(input_filename)
        print(open(tmp_path / BASENAME, encoding='utf-8').read())


if __name__ == '__main__':
    # Emulate pytest / fixtures
    from pathlib import Path

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for module in ['ome_zarr', 'zarr', 'numcodecs', 'asyncio']:
        logging.getLogger(module).setLevel(logging.WARNING)

    test = TestRocrate()
    for input_filename in test.input_filenames:
        for output_format in test.output_formats:
            test.test_rocrate(Path(tempfile.TemporaryDirectory().name), input_filename)
