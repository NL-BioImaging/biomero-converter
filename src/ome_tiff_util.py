from ome_types.model import *
from ome_types import to_xml


def create_metadata(source):
    ome = OME()

    return to_xml(ome)


def create_resolution_data(source):
    pixel_size_um = source.get_pixel_size()
    resolution_unit = 'CENTIMETER'
    resolution = [1e4 / size for size in pixel_size_um]
    return resolution, resolution_unit
