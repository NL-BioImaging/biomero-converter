import numpy as np
from ome_types._mixins._ids import ID_COUNTER
from ome_types.model import *
from ome_types import to_xml
from tifffile import xml2dict
import uuid

from src.color_conversion import rgba_to_int, int_to_rgba
from src.parameters import VERSION
from src.util import *


def metadata_to_dict(xml_metadata):
    metadata = xml2dict(xml_metadata)
    if 'OME' in metadata:
        metadata = metadata['OME']
    return metadata


def create_uuid():
    return f'urn:uuid:{uuid.uuid4()}'


def reset_ome_ids():
    ID_COUNTER.clear()   # this will reset all reference/ids


def read_ome_xml_metadata(metadata):
    pixel_size = {}
    position = {}
    channels = []
    rows = set()
    columns = set()
    fields = set()
    wells = {}
    image_refs = {}

    image0 = ensure_list(metadata.get('Image', []))[0]
    is_plate = 'Plate' in metadata
    if is_plate:
        plate = metadata['Plate']
        name = plate.get('Name')
        for well in ensure_list(plate['Well']):
            row = create_row_col_label(well['Row'], plate['RowNamingConvention'])
            column = create_row_col_label(well['Column'], plate['ColumnNamingConvention'])
            rows.add(row)
            columns.add(column)
            label = f'{row}{column}'
            wells[label] = well['ID']
            image_refs[label] = {}
            for sample in ensure_list(well.get('WellSample')):
                sample_id_parts = sample['ID'].split(':')
                field_id = sample_id_parts[-1]
                fields.add(int(field_id))
                image_refs[label][field_id] = sample['ImageRef']['ID']
        if 'Rows' in plate:
            rows = [create_row_col_label(row, plate['RowNamingConvention']) for row in range(plate['Rows'])]
        else:
            rows = sorted(rows)
        if 'Columns' in plate:
            columns = [create_row_col_label(col, plate['ColumnNamingConvention']) for col in
                            range(plate['Columns'])]
        else:
            columns = sorted(columns, key=int)
        wells = list(wells.keys())
        fields = sorted(fields)
        image_refs = image_refs
    else:
        name = image0.get('Name')
    acquisition_datetime = image0.get('AcquisitionDate')
    pixels = image0.get('Pixels', {})
    dtype0 = pixels['Type'].lower()
    if dtype0 in ['float', 'double']:
        dtype0 = 'float64' if dtype0 == 'double' else 'float32'
    dtype = np.dtype(dtype0)
    if 'PhysicalSizeX' in pixels:
        pixel_size['x'] = convert_to_um(float(pixels.get('PhysicalSizeX')), pixels.get('PhysicalSizeXUnit'))
    if 'PhysicalSizeY' in pixels:
        pixel_size['y'] = convert_to_um(float(pixels.get('PhysicalSizeY')), pixels.get('PhysicalSizeYUnit'))
    if 'PhysicalSizeZ' in pixels:
        pixel_size['z'] = convert_to_um(float(pixels.get('PhysicalSizeZ')), pixels.get('PhysicalSizeZUnit'))
    plane = pixels.get('Plane')
    if plane:
        if 'PositionX' in plane:
            position['x'] = convert_to_um(float(plane.get('PositionX')), plane.get('PositionXUnit'))
        if 'PositionY' in plane:
            position['y'] = convert_to_um(float(plane.get('PositionY')), plane.get('PositionYUnit'))
        if 'PositionZ' in plane:
            position['z'] = convert_to_um(float(plane.get('PositionZ')), plane.get('PositionZUnit'))
    for channel0 in ensure_list(pixels.get('Channel')):
        channel = {}
        if 'Name' in channel0:
            channel['label'] = channel0['Name']
        if 'Color' in channel0:
            channel['color'] = int_to_rgba(channel0['Color'])
        for key, value in channel0.items():
            if key not in ['Name', 'Color'] and value is not None:
                channel[camel_to_snake(key)] = value
        channels.append(channel)
    if 'SignificantBits' in pixels:
        bits_per_pixel = int(pixels['SignificantBits'])
    else:
        bits_per_pixel = dtype.itemsize * 8

    microscope_info = camel_to_snake_keys_dict(metadata.get('Instrument'))
    microscope_info.update(microscope_info.pop('objective', {}))

    return (name, is_plate, pixel_size, position, dtype, bits_per_pixel, channels, microscope_info, acquisition_datetime,
            wells, list(rows), list(columns), list(fields), image_refs)


def create_metadata(source, dim_order='tczyx', uuid=None, image_uuids=None, image_filenames=None, wells=None,
                    metadata_only=False):
    ome = OME()
    if uuid is None:
        uuid = create_uuid()
    ome.uuid = uuid
    ome.creator = f'nl.biomero.OmeTiffWriter {VERSION}'

    microscope_info = source.get_microscope_info()
    instrument_id = None
    objective_id = None
    if microscope_info:
        microscope = Microscope()
        has_microscope = False
        manufacturer = microscope_info.get('manufacturer')
        if manufacturer is not None:
            microscope.manufacturer = manufacturer
            has_microscope = True
        model = microscope_info.get('model')
        if model is not None:
            microscope.model = model
            has_microscope = True
        serial_number = microscope_info.get('serial_number')
        if serial_number is not None:
            microscope.serial_number = serial_number
            has_microscope = True

        objective = Objective()
        has_objective = False
        magnification = microscope_info.get('magnification',
                                            microscope_info.get('nominal_magnification',
                                                                microscope_info.get('NominalMagnification')))
        if magnification is not None:
            objective.nominal_magnification = magnification
            has_objective = True
        lens_na = microscope_info.get('n_a', microscope_info.get('lens_na'))
        if lens_na is not None:
            objective.lens_na = lens_na
            has_objective = True

        if has_microscope or has_objective:
            instrument = Instrument()
            instrument_id = instrument.id
            if has_microscope:
                instrument.microscope = microscope
            if has_objective:
                instrument.objectives.append(objective)
            ome.instruments = [instrument]

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
                image_uuid = image_uuids[image_index] if image_uuids is not None else None
                image_filename = image_filenames[image_index] if image_filenames is not None else None
                image = create_image_metadata(source,
                                              image_name,
                                              dim_order,
                                              image_uuid,
                                              image_filename,
                                              instrument_id=instrument_id,
                                              objective_id=objective_id,
                                              metadata_only=metadata_only)
                ome.images.append(image)

                image_ref = ImageRef(id=image.id)   # assign id at instantiation to avoid auto sequence increment
                sample.image_ref = image_ref
                well.well_samples.append(sample)

                image_index += 1

            plate.wells.append(well)

        ome.plates = [plate]
    else:
        image_filename0 = image_filenames[0] if image_filenames is not None else None
        ome.images = [
            create_image_metadata(source, source.get_name(), dim_order, ome.uuid, image_filename0,
                                  instrument_id=instrument_id, objective_id=objective_id,
                                  metadata_only=metadata_only)
        ]

    return to_xml(ome)


def create_image_metadata(source, image_name, dim_order='tczyx', image_uuid=None, image_filename=None,
                          instrument_id=None, objective_id=None, metadata_only=False):
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
            acquisition_mode = channel.get('acquisition_mode', channel.get('AcquisitionMode'))
            if acquisition_mode:
                ome_channel.acquisition_mode = acquisition_mode
            emission_wavelength = channel.get('emission_wavelength', channel.get('EmissionWavelength'))
            if emission_wavelength is not None:
                ome_channel.emission_wavelength = emission_wavelength
                ome_channel.emission_wavelength_unit = channel.get('emission_wavelength_unit', UnitsLength.NANOMETER)
            excitation_wavelength = channel.get('excitation_wavelength', channel.get('ExcitationWavelength'))
            if excitation_wavelength is not None:
                ome_channel.excitation_wavelength = excitation_wavelength
                ome_channel.excitation_wavelength_unit = channel.get('excitation_wavelength_unit', UnitsLength.NANOMETER)
            pinhole_size = channel.get('pinhole_size', channel.get('PinholeSize'))
            if pinhole_size is not None:
                ome_channel.pinhole_size = pinhole_size
                pinhole_size_unit = channel.get('pinhole_size_unit', channel.get('PinholeSizeUnit'))
                if pinhole_size_unit:
                    ome_channel.pinhole_size_unit = pinhole_size_unit

            ome_channels.append(ome_channel)

    pixel_type = str(source.get_dtype())
    if pixel_type.startswith('float'):
        pixel_type = PixelType.DOUBLE if '64' in pixel_type else PixelType.FLOAT
    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder(dim_order[::-1].upper()),
        type=PixelType(pixel_type),
        channels=ome_channels,
        size_t=t, size_c=c, size_z=z, size_y=y, size_x=x,
    )
    if metadata_only:
        pixels.metadata_only = MetadataOnly()
    elif image_uuid:
        tiff_data = TiffData()
        tiff_data.uuid = TiffData.UUID(value=image_uuid, file_name=image_filename)
        pixels.tiff_data_blocks=[tiff_data]

    if 'x' in pixel_size:
        pixels.physical_size_x = pixel_size['x']
        pixels.physical_size_x_unit = UnitsLength.MICROMETER
    if 'y' in pixel_size:
        pixels.physical_size_y = pixel_size['y']
        pixels.physical_size_y_unit = UnitsLength.MICROMETER
    if 'z' in pixel_size:
        pixels.physical_size_z = pixel_size['z']
        pixels.physical_size_z_unit = UnitsLength.MICROMETER
    significant_bits = source.get_significant_bits()
    if significant_bits:
        pixels.significant_bits = significant_bits

    image = Image(name=image_name, pixels=pixels)
    acquisition_datetime = source.get_acquisition_datetime()
    if acquisition_datetime:
        image.acquisition_date = acquisition_datetime
    index = pixels.id.split(':')[1]
    for channeli, channel in enumerate(pixels.channels):
        channel.id = f'Channel:{index}:{channeli}'
    if instrument_id is not None:
        image.instrument_ref = InstrumentRef(id=instrument_id)
    if objective_id is not None:
        image.objective_settings = ObjectiveSettings(id=objective_id)
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
