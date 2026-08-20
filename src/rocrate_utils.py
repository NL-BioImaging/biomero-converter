# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py

from datetime import datetime
from rocrate.model import ContextEntity

from src.util import flatten_dict
from src.zarr_extension import ZarrCrate


def create_ro_crate(source, dest_path={}):
    crate = ZarrCrate()
    # Alternative use github German-BioImaging idr_study_crates GraphBuilder class to low-level build instead?

    properties = {}
    properties['name'] = source.get_name()  # use output path(s) instead
    properties['encodingFormat'] = [
        'application/vnd.zarr',
        {'@id': 'https://openminds.docs.om-i.org/en/v3.0/instance_libraries/contentTypes.html#application-vnd-zarr'}
    ]
    #properties["description"] = source.get_description()
    #properties["license"] = source.get_license()
    dataset_entity = crate.add_dataset(dest_path='.', properties=properties)

    additional_properties = []
    for index, (key, value) in enumerate(flatten_dict(source.get_acquisition_metadata()).items()):
        if isinstance(value, datetime):
            value = str(value)
        additional_properties.append({
            '@id': f'#acq:{index:03d}',
            '@type': 'PropertyValue',
            'name': key,
            'value': value
        })

    properties_entities = []
    for additional_property in additional_properties:
        properties_entity = ContextEntity(crate, identifier=additional_property['@id'], properties=additional_property)
        properties_entities.append(crate.add(properties_entity))

    instrument_properties = {
        '@id': '#microscope-001',
        '@type': 'IndividualProduct',
    }

    metadata = source.get_metadata()

    instrument_name = search_metadata_fully(metadata, ['model', 'name', 'identifier'],
                                            contexts=['instrument', 'microscope', 'device', ''])
    if instrument_name:
        instrument_properties['name'] = instrument_name

    manufacturer = search_metadata_fully(metadata, ['manufacturer', 'make'],
                                         contexts=['instrument', 'microscope', 'device', ''])
    if manufacturer:
        instrument_properties['manufacturer'] = manufacturer

    serial_number = search_metadata_fully(metadata, ['serialnumber', 'serial'],
                                         contexts=['instrument', 'microscope', 'device', ''])
    if serial_number:
        instrument_properties['serialNumber'] = serial_number

    instrument_entity = ContextEntity(crate, identifier=instrument_properties['@id'], properties=instrument_properties)
    instrument_entity['additionalProperty'] = properties_entities
    create_entity = crate.add_action(instrument_entity, identifier='#data-capture-001')
    create_entity['instrument'] = instrument_entity
    create_entity['result'] = dataset_entity

    crate.add(instrument_entity)

    # TODO: Can add variableMeasured for output properties

    crate.write(dest_path)
    return crate


def search_metadata_fully(metadata, labels, contexts=None):
    for context in contexts:
        for label in labels:
            search_labels = [label]
            if context:
                search_labels.append(context)
            value = search_metadata(metadata, search_labels)
            if value is not None:
                return value
    return None


def search_metadata(metadata, labels):
    for key, value in metadata.items():
        if isinstance(value, dict):
            match = search_metadata(value, labels)
            if match is not None:
                return match
        else:
            key1 = key.lower()
            for label in labels:
                label1 = label.lower()
                if label1 in key1 and not isinstance(value, dict):
                    return value
    return None
