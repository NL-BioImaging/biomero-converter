# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py

from datetime import datetime
from imaging_metadata_converter import SOURCE_MAP_KEY
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

    # acquisition metadata on the common imaging metadata model, without the map of where each field came from
    model_metadata = {key: value for key, value in source.get_model_metadata().items() if key != SOURCE_MAP_KEY}
    model_instrument = model_metadata.get('Instrument', {})

    additional_properties = []
    for index, (key, value) in enumerate(flatten_dict(model_metadata).items()):
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
    # fallback: search the decoded acquisition metadata first, then the source metadata
    searchable = {'acquisition': source.get_acquisition_metadata(), 'source': metadata}

    uri = source.uri

    if model_instrument.get('Manufacturer'):
        manufacturer = model_instrument['Manufacturer']
    elif 'manufacturer' in metadata and metadata['manufacturer']:
        manufacturer = metadata['manufacturer']
    else:
        manufacturer = search_metadata_fully(searchable, ['manufacturer', 'make'],
                                             contexts=['instrument', 'microscope', 'device', 'system', ''])
    if manufacturer:
        instrument_properties['manufacturer'] = manufacturer

    if model_instrument.get('Model') or model_instrument.get('Name'):
        model = model_instrument.get('Model') or model_instrument['Name']
    elif 'model' in metadata and metadata['model']:
        model = metadata['model']
    else:
        model = search_metadata_fully(searchable, ['model', 'name', 'product', 'productname', 'identifier'],
                                      contexts=['instrument', 'microscope', 'device', 'system', ''])
    if model:
        instrument_properties['name'] = model

    if 'serial' in metadata and metadata['serial']:
        serial = metadata['serial']
    else:
        serial = search_metadata_fully(searchable, ['serialnumber', 'serial'],
                                       contexts=['instrument', 'microscope', 'device', 'system', ''])
    if serial:
        instrument_properties['serialNumber'] = serial

    instrument_entity = ContextEntity(crate, identifier=instrument_properties['@id'], properties=instrument_properties)
    instrument_entity['additionalProperty'] = properties_entities
    create_entity = crate.add_action(instrument_entity, identifier='#data-capture-001')
    create_entity['instrument'] = instrument_entity
    create_entity['result'] = dataset_entity

    crate.add(instrument_entity)

    # TODO: Can add variableMeasured for output properties - or link to external file e.g. csv

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
