# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py

from io import StringIO
import os.path
import re
import yaml
from imaging_metadata_converter import SOURCE_MAP_KEY
from imaging_metadata_converter.ModelPaths import DEFAULT_MODEL_FILE
from rocrate.model import ContextEntity

from src.util import to_plain_types
from src.zarr_extension import ZarrCrate


ACQUISITION_METADATA_FILENAME = 'acquisition_metadata.yaml'
MODEL_SCHEMA_URL = ('https://github.com/NL-BioImaging/imaging-metadata-converter/blob/{release}/src/'
                    'imaging_metadata_converter/models/' + os.path.basename(DEFAULT_MODEL_FILE))


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

    # acquisition metadata on the common imaging metadata model, written to a yaml file the crate points to
    model_metadata = {key: value for key, value in source.get_model_metadata().items() if key != SOURCE_MAP_KEY}
    # given as the file's source, so the crate writes the file itself
    # libyaml's dumper where installed: a Leica tile scan's metadata is tens of thousands of values
    yaml_text = yaml.dump(to_plain_types(model_metadata), Dumper=getattr(yaml, 'CSafeDumper', yaml.SafeDumper),
                          sort_keys=False, allow_unicode=True)
    acquisition_metadata_entity = crate.add_file(
        source=StringIO(yaml_text), dest_path=ACQUISITION_METADATA_FILENAME, properties={
        'name': 'Acquisition metadata',
        'description': 'Image acquisition metadata converted to the imaging metadata model '
                       '(https://github.com/NL-BioImaging/imaging-metadata-converter)',
        'encodingFormat': 'application/yaml',
    })
    # the dataset entity replaced the crate's root dataset, so link the file to it explicitly
    dataset_entity.append_to('hasPart', acquisition_metadata_entity)

    model_instrument = model_metadata.get('Instrument', {})

    instrument_properties = {
        '@id': '#microscope-001',
        '@type': 'IndividualProduct',
    }

    metadata = source.get_metadata()
    # fallback: search the decoded acquisition metadata first, then the source metadata
    searchable = {'acquisition': source.get_acquisition_metadata(), 'source': metadata}

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
        # schema.org serialNumber is text
        instrument_properties['serialNumber'] = str(serial)

    instrument_entity = ContextEntity(crate, identifier=instrument_properties['@id'], properties=instrument_properties)
    create_entity = crate.add_action(instrument_entity, identifier='#data-capture-001')
    create_entity['instrument'] = instrument_entity
    create_entity['result'] = dataset_entity
    try:
        start_time = source.get_acquisition_datetime()
    except NotImplementedError:
        start_time = None
    if start_time:
        create_entity['startTime'] = to_iso_datetime(start_time)
    end_time = source.get_acquisition_end_datetime()
    if end_time:
        create_entity['endTime'] = to_iso_datetime(end_time)
    # list the action on the root, so it can be found from there
    dataset_entity.append_to('mentions', create_entity)

    crate.add(instrument_entity)

    # the metadata file describes the acquisition, on the model's LinkML schema
    acquisition_metadata_entity['about'] = create_entity
    schema_properties = {
        '@type': ['CreativeWork', 'Profile'],
        'name': 'Imaging metadata model (LinkML schema)',
        'encodingFormat': 'application/yaml',
    }
    schema_version = get_model_schema_version()
    if schema_version:
        schema_properties['version'] = schema_version
    # the model's release is tagged with its version
    schema_url = MODEL_SCHEMA_URL.format(release=f'v{schema_version}' if schema_version else 'main')
    schema_entity = crate.add(ContextEntity(crate, identifier=schema_url, properties=schema_properties))
    acquisition_metadata_entity['conformsTo'] = schema_entity

    # TODO: Can add variableMeasured for output properties

    crate.write(dest_path)
    return crate


def to_iso_datetime(value):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def get_model_schema_version():
    # the version is in the schema's header: read up to it rather than loading the whole (large) schema
    with open(DEFAULT_MODEL_FILE, encoding='utf-8') as file:
        for line in file:
            match = re.match(r'version:\s*(\S+)', line)
            if match:
                return match.group(1).strip('\'"')
    return None


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
