"""The metadata a conversion writes, read from the sources without their pixel data, so it runs in seconds.

For each local example: the source metadata passed to the imaging metadata converter (all of it, but what only
lays out the pixel data), the model metadata it becomes, the OME-XML of the output and the RO-Crate's YAML.

Also converts the tiff examples to OME-Zarr v3 with its RO-Crate, and copies each acquisition metadata YAML
into metadata_output/ in this repo, to compare and review.
"""

import glob
import json
import os
import re
import shutil
import sys

from imaging_metadata_converter import convert_metadata
import pytest
import yaml

sys.path.append(os.getcwd())

from converter import _convert
from src.DicomSource import DicomSource
from src.helper import create_source
from src.ome_tiff_util import create_metadata
from src.rocrate_utils import ACQUISITION_METADATA_FILENAME, create_ro_crate
from src.util import to_plain_types


SLIDES = 'C:/Project/slides/'
METADATA_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'metadata_output')
# smallest first; the OME-XML examples have no pixel data to convert
CONVERT_FILENAMES = sorted(glob.glob(SLIDES + 'tiff/*.tif*'), key=os.path.getsize)

# what only lays out or encodes the pixel data, and the path on this disk: never in the source metadata
LAYOUT_KEYS = {'filepath', 'BytesInc', 'BitInc', 'StripOffsets', 'StripByteCounts', 'TileOffsets', 'TileByteCounts',
               'JPEGTables', 'RowsPerStrip', 'Compression', 'TiffData', 'BinData', 'BigEndian', 'PixelData'}


def axial_ct_series():
    # 2_skull_ct opens its scout by default; its 67-slice axial stack is the one with a slice spacing
    series = DicomSource.get_series(SLIDES + 'dicom/2_skull_ct') if os.path.exists(SLIDES + 'dicom/2_skull_ct') else {}
    return next((uid for uid, filenames in series.items() if len(filenames) == 67), None)


# per example: its source arguments, model metadata values (dotted paths) and OME-XML fragments
EXAMPLES = {
    'Leica/RGB.lif': ({}, {
        'Instrument.Model': 'MICA', 'Objective.Magnification': 10, 'Objective.ImmersionType': 'Air',
        'Objective.LensNA': 0.32, 'GenericExcitationSource': [{'ID': 'LightSource:0', 'Role': ['Transmitted']}],
        'Plane.ExposureTime': 0.04, 'Plane.ExposureTimeUnit': 's'},
        ['Immersion="Air" LensNA="0.32" NominalMagnification="10.0"']),
    'Leica/TileScan.lof': ({}, {
        'Instrument.Model': 'DMI6000B-CS', 'Objective.ImmersionType': 'Oil', 'Objective.LensNA': 1.3,
        'Pixels.Channel[0].Name': 'Leica/ALEXA 488', 'Pixels.Channel[1].Name': 'Leica/mCherry',
        'Pixels.Channel[2].Name': 'Leica/Cerulean', 'Pixels.Channel[0].Fluorophore.ExcitationWavelength': 488,
        'Laser[3].Role': ['Fluorescence']},
        ['Name="ALEXA 488"', 'Name="mCherry"', 'Name="Cerulean"', 'Immersion="Oil" LensNA="1.3"']),
    'Leica/SP5-3D-Confocal_Timeseries.lif': ({}, {
        'Instrument.Name': 'TCS SP5', 'Objective.Magnification': 63.0, 'Objective.ImmersionType': 'Water',
        'Objective.LensNA': 0.9, 'Pixels.SizeZ': 32, 'Pixels.SizeT': 61, 'Pixels.PhysicalSizeZ': 2.964e-07},
        ['Immersion="Water" LensNA="0.9" NominalMagnification="63.0"']),
    'Leica/LMD7-Test.lif': ({}, {
        'Laser.Model': 'Explorer', 'Laser.Role': ['Microdissection'], 'Plane.ExposureTime': 0.083,
        'AcquisitionSoftware.Version': '8.5.9136', 'Image.AcquisitionDate': '2025-07-10T14:07:11.516173'}, []),
    'tiff/EMSIS Xarosa.tif': ({}, {
        'Instrument.Manufacturer': 'EMSIS', 'Plane.ExposureTime': 0.0082, 'Plane.ExposureTimeUnit': 's',
        'Image.AcquisitionDate': '2025-05-28 10:54:00', 'XResolution': 72.0}, []),
    'tiff/Cikteq SEM4000x Normal.tiff': ({}, {
        'Plane.PixelDwellTime': 5.0, 'Plane.PixelDwellTimeUnit': 'µs', 'Image.ScanSettings.FrameTime.Value': 149,
        'Image.ScanSettings.FrameTime.Unit': 's'}, []),
    'tiff/TFS Phenom Pharos.tiff': ({}, {
        'Instrument.Model': 'Phenom Pharos G2', 'Instrument.CatalogNumber': 'MVE084613-20046-F'}, []),
    'tiff/DNAcropSmall.ome.tiff': ({}, {
        'Objective.ImmersionType': 'Oil', 'Objective.LensNA': 1.4, 'Objective.Magnification': 100.0},
        ['Immersion="Oil" LensNA="1.4" NominalMagnification="100.0"']),
    'dicom/2_skull_ct': ({'image_uuid': axial_ct_series()}, {
        'Instrument.Manufacturer': 'GE MEDICAL SYSTEMS', 'Pixels.PhysicalSizeZ': 2.5, 'Pixels.PhysicalSizeZUnit': 'mm',
        'Plane.ExposureTime': 800, 'Plane.ExposureTimeUnit': 'ms'}, []),
    'dicom/MRBRAIN.DCM': ({}, {
        'Instrument.Manufacturer': 'Philips', 'Pixels.PhysicalSizeZ': 7.5, 'Pixels.PhysicalSizeZUnit': 'mm'}, []),
}


def value_at(metadata, path):
    """The value at a dotted path, list items as "[i]", or a KeyError naming what is missing."""
    node = metadata
    for key in path.split('.'):
        match = re.fullmatch(r'(.+?)((?:\[\d+\])*)', key)
        node = node[match.group(1)]
        for index in re.findall(r'\[(\d+)\]', match.group(2)):
            node = node[int(index)]
    return node


def keys_anywhere(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from keys_anywhere(value)
    elif isinstance(node, list):
        for item in node:
            yield from keys_anywhere(item)


@pytest.mark.parametrize('name', EXAMPLES)
def test_metadata(tmp_path, name):
    path = SLIDES + name
    if not os.path.exists(path):
        pytest.skip(f'{path} is not here')
    source_kwargs, expected, ome_fragments = EXAMPLES[name]
    source = create_source(path, **source_kwargs)
    source.init_metadata()

    source_metadata = to_plain_types(source.get_source_metadata())
    assert not LAYOUT_KEYS & set(keys_anywhere(source_metadata))

    # the converter reads the source as it reads its JSON example: the rules are written against those
    model = source.get_model_metadata()
    assert model == convert_metadata(json.loads(json.dumps(source_metadata, default=str)))
    for model_path, value in expected.items():
        assert value_at(model, model_path) == value, model_path

    xml = create_metadata(source, metadata_only=True)
    xml = xml[0] if isinstance(xml, tuple) else xml
    for fragment in ome_fragments:
        assert fragment in xml

    create_ro_crate(source, str(tmp_path))
    with open(tmp_path / ACQUISITION_METADATA_FILENAME, encoding='utf-8') as file:
        crate_metadata = yaml.load(file, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    assert 'SourceMap' not in crate_metadata
    for model_path, value in expected.items():
        assert value_at(crate_metadata, model_path) == value, model_path


@pytest.mark.parametrize('input_filename', CONVERT_FILENAMES, ids=os.path.basename)
def test_converted_metadata(tmp_path, input_filename):
    output = json.loads(_convert(input_filename, str(tmp_path), output_format='omezarr05'))
    yaml_filename = os.path.join(output[0]['full_path'], ACQUISITION_METADATA_FILENAME)
    with open(yaml_filename, encoding='utf-8') as file:
        crate_metadata = yaml.load(file, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    assert crate_metadata
    assert 'SourceMap' not in crate_metadata

    os.makedirs(METADATA_OUTPUT, exist_ok=True)
    shutil.copy2(yaml_filename, os.path.join(METADATA_OUTPUT, os.path.basename(input_filename) + '.yaml'))
