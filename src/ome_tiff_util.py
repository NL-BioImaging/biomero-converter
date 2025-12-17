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


def create_metadata(source, dim_order='tczyx', uuid=None, image_uuids=None, image_filenames=None, wells=None):
    ome = OME()
    if uuid is None:
        uuid = create_uuid()
    ome.uuid = uuid
    ome.creator = f'nl.biomero.OmeTiffWriter {VERSION}'

    if source.is_screen():
        if wells is None:
            wells = source.get_wells()

        nrows, row_type = get_row_col_len_type(source.get_rows())
        ncols, col_type = get_row_col_len_type(source.get_columns())

        plate = Plate()
        plate.name = source.get_name()
        plate.rows = nrows
        plate.columns = ncols
        plate.row_naming_convention = row_type
        plate.column_naming_convention = col_type

        image_index = 0
        for well_id in wells:
            row, col = split_well_name(well_id)
            row_index = get_row_col_index(row)
            col_index = get_row_col_index(col)
            well = Well(row=row_index, column=col_index)
            well.id = f'Well:{row_index}:{col_index}'
            for field in source.get_fields():
                sample = WellSample(index=image_index)
                sample.id = f'WellSample:{row_index}:{col_index}:{field}'
                position = source.get_position_um(well_id)
                if 'x' in position:
                    sample.position_x = position['x']
                    sample.position_x_unit = UnitsLength.MICROMETER
                if 'y' in position:
                    sample.position_y = position['y']
                    sample.position_y_unit = UnitsLength.MICROMETER

                image_name = f'Well {well_id}, Field #{int(field) + 1}'
                image = create_image_metadata(source,
                                              image_name,
                                              dim_order,
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
        ome.images = [create_image_metadata(source, source.get_name(), dim_order, ome.uuid, image_filenames[0])]

    return to_xml(ome)


def create_image_metadata(source, image_name, dim_order='tczyx', image_uuid=None, image_filename=None):
    t, c, z, y, x = [source.get_shape()[source.dim_order.index(dim)] if dim in source.get_dim_order() else 1
                     for dim in 'tczyx']
    pixel_size = source.get_pixel_size_um()
    channels = source.get_channels()
    if source.is_rgb():
        ome_channels = [Channel(name='rgb', samples_per_pixel=3)]
    elif len(channels) < c:
        ome_channels = [Channel(name=f'{channeli}', samples_per_pixel=1) for channeli in range(c)]
    else:
        ome_channels = []
        for channeli, channel in enumerate(channels):
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

    image = Image(name=image_name, pixels=pixels)
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


def get_row_col_len_type(labels):
    max_index = max(get_row_col_index(label) for label in labels)
    nlen = max_index + 1
    is_digits = [label.isdigit() for label in labels]
    if np.all(is_digits):
        naming_convention = NamingConvention.NUMBER
    else:
        naming_convention = NamingConvention.LETTER
    return nlen, naming_convention


def get_row_col_index(label):
    if label.isdigit():
        index = int(label) - 1
    else:
        index = ord(label.upper()) - ord('A')
    return index


def create_row_col_label(index, naming_convention):
    if naming_convention.lower() == NamingConvention.LETTER.name.lower():
        label = chr(ord('A') + index)
    else:
        label = index + 1
    return str(label)


def create_resolution_metadata(source):
    pixel_size_um = source.get_pixel_size_um()
    resolution_unit = 'CENTIMETER'
    resolution = [1e4 / pixel_size_um[dim] for dim in 'xy']
    return resolution, resolution_unit
