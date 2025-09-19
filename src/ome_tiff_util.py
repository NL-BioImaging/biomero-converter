import numpy as np
from ome_types.model import *
from ome_types import to_xml
from tifffile import xml2dict
import uuid

from src.color_conversion import rgba_to_int
from src.parameters import VERSION
from src.util import split_well_name


def metadata_to_dict(xml_metadata):
    metadata = xml2dict(xml_metadata)
    if 'OME' in metadata:
        metadata = metadata['OME']
    return metadata


def create_uuid():
    return f'urn:uuid:{uuid.uuid4()}'


def create_metadata(source, uuid=None, image_uuids=None, image_filenames=None):
    ome = OME()
    if uuid is None:
        uuid = create_uuid()
    ome.uuid = uuid
    ome.creator = f'nl.biomero.OmeTiffWriter {VERSION}'

    if source.is_screen():
        columns = source.get_columns()
        rows = source.get_rows()

        plate = Plate()
        plate.columns = len(columns)
        plate.rows = len(rows)
        plate.row_naming_convention = get_col_row_type(rows)
        plate.column_naming_convention = get_col_row_type(columns)

        image_index = 0
        for well_id in source.get_wells():
            row, col = split_well_name(well_id)
            col_index = columns.index(col)
            row_index = rows.index(row)
            well = Well(column=col_index, row=row_index)
            well.id = f'Well:{well_id}'
            for field in source.get_fields():
                sample = WellSample(index=image_index)
                sample.id = f'WellSample:{well_id}:{field}'
                position = source.get_position_um(well_id)
                if 'x' in position:
                    sample.position_x = position['x']
                    sample.position_x_unit = UnitsLength.MICROMETER
                if 'y' in position:
                    sample.position_y = position['y']
                    sample.position_y_unit = UnitsLength.MICROMETER

                image = create_image_metadata(source,
                                              image_uuids[image_index],
                                              image_filenames[image_index])
                ome.images.append(image)

                image_ref = ImageRef(id=image.id)   # assign id at instantiation to avoid auto sequence increment
                sample.image_ref = image_ref
                well.well_samples.append(sample)

                image_index += 1

            plate.wells.append(well)

        ome.plates = [plate]
    else:
        ome.images = [create_image_metadata(source, ome.uuid, source.get_name())]

    return to_xml(ome)


def create_image_metadata(source, image_uuid=None, image_filename=None):
    dim_order = 'tczyx'
    t, c, z, y, x = source.get_shape()
    pixel_size = source.get_pixel_size_um()
    ome_channels = []
    for channeli, channel in enumerate(source.get_channels()):
        ome_channel = Channel()
        ome_channel.name = channel.get('label', channel.get('Name', f'{channeli}'))
        ome_channel.samples_per_pixel = 1
        color = channel.get('color', channel.get('Color'))
        if color is not None:
            ome_channel.color = Color(rgba_to_int(color))
        ome_channels.append(ome_channel)

    tiff_data = TiffData()
    tiff_data.uuid = TiffData.UUID(value=image_uuid, file_name=image_filename)

    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder(dim_order[::-1].upper()),
        type=PixelType(str(source.get_dtype())),
        channels=ome_channels,
        size_t=t, size_c=c, size_z=z, size_y=y, size_x=x,
        tiff_data_blocks=[tiff_data]
    )
    if 'x' in pixel_size:
        pixels.physical_size_x = pixel_size['x']
        pixels.physical_size_x_unit = UnitsLength.MICROMETER
    if 'y' in pixel_size:
        pixels.physical_size_y = pixel_size['y']
        pixels.physical_size_y_unit = UnitsLength.MICROMETER
    if 'z' in pixel_size:
        pixels.physical_size_z = pixel_size['z']
        pixels.physical_size_z_unit = UnitsLength.MICROMETER

    image = Image(pixels=pixels)
    index = pixels.id.split(':')[1]
    for channeli, channel in enumerate(pixels.channels):
        channel.id = f'Channel:{index}:{channeli}'
    return image


def create_binaryonly_metadata(metadata_filename, companion_uuid):
    ome = OME()
    ome.uuid = create_uuid()
    ome.creator = f'nl.biomero.OmeTiffWriter {VERSION}'
    ome.binary_only = OME.BinaryOnly(metadata_file=metadata_filename, uuid=companion_uuid)
    return to_xml(ome), ome.uuid


def get_col_row_type(labels):
    is_digits = [label.isdigit() for label in labels]
    if np.all(is_digits):
        naming_convention = NamingConvention.NUMBER
    else:
        naming_convention = NamingConvention.LETTER
    return naming_convention


def create_col_row_label(index, naming_convention):
    label = index + 1
    if naming_convention.lower() == NamingConvention.LETTER.name.lower():
        label = chr(ord('A') + index)
    return str(label)


def create_resolution_metadata(source):
    pixel_size_um = source.get_pixel_size_um()
    resolution_unit = 'CENTIMETER'
    resolution = [1e4 / pixel_size_um[dim] for dim in 'xy']
    return resolution, resolution_unit
