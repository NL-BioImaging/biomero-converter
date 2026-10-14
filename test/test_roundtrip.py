"""Synthetic round trips (no test data needed): TIFF -> OME-Zarr -> OME-TIFF keeps names and pixels."""
import json

import numpy as np
import pytest
import tifffile

from biomero_converter.converter import _convert
from biomero_converter.helper import create_source

ZARR_FORMATS = ['omezarr04', 'omezarr05']


def _convert_one(input_filename, output_folder, output_format):
    [result] = json.loads(_convert(str(input_filename), str(output_folder), output_format=output_format))
    return result


def _read(filename):
    source = create_source(str(filename))
    source.init_metadata()
    return source, np.squeeze(source.get_data(source.get_dim_order()))


def _images():
    rng = np.random.default_rng(0)
    return {
        # names ending in 'e'/'m'/'o' must survive (no rstrip('.ome'))
        'sample': (rng.integers(0, 256, (64, 80, 3), dtype=np.uint8), 'YXS'),
        # 16-bit 3-channel fluorescence: channels, not RGB
        'fluo_image': (rng.integers(0, 4096, (3, 64, 80), dtype=np.uint16), 'CYX'),
    }


@pytest.mark.parametrize('name', list(_images()))
@pytest.mark.parametrize('zarr_format', ZARR_FORMATS)
def test_tiff_zarr_tiff_round_trip(tmp_path, name, zarr_format):
    data, axes = _images()[name]
    tiff = tmp_path / f'{name}.ome.tif'
    tifffile.imwrite(tiff, data, photometric='rgb' if axes.endswith('S') else 'minisblack',
                     ome=True, metadata={'axes': axes, 'Name': name})

    zarr_result = _convert_one(tiff, tmp_path / 'zarr', zarr_format)
    assert zarr_result['name'] == name
    assert zarr_result['full_path'].endswith(f'{name}.ome.zarr')

    tiff_result = _convert_one(zarr_result['full_path'], tmp_path / 'tiff', 'ometiff')
    assert tiff_result['name'] == name
    assert tiff_result['full_path'].endswith(f'{name}.ome.tiff')

    _, original = _read(tiff)
    _, round_trip = _read(tiff_result['full_path'])
    np.testing.assert_array_equal(round_trip, original)


def test_zarr_without_omero_metadata_to_tiff(tmp_path):
    data = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80)
    tiff = tmp_path / 'plain.tif'
    tifffile.imwrite(tiff, data)
    zarr_result = _convert_one(tiff, tmp_path / 'zarr', 'omezarr04')
    tiff_result = _convert_one(zarr_result['full_path'], tmp_path / 'tiff', 'ometiff')
    _, round_trip = _read(tiff_result['full_path'])
    np.testing.assert_array_equal(round_trip, data)
