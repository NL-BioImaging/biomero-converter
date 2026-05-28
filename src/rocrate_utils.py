# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py


from rocrate.model import ContextEntity, ComputationalWorkflow

from src.rembi_extension import ImageAcquisition
from src.zarr_extension import ZarrCrate


def create_ro_crate(source, dest_path={}):
    crate = ZarrCrate()
    # TODO: ContextEntity sub-classes needed for each type with verbose code, while essentially same functionality
    # TODO: Alternative use github German-BioImaging idr_study_crates GraphBuilder class to low-level build instead?

    properties = {}
    properties['name'] = source.get_name()
    #properties["description"] = source.get_description()
    #properties["license"] = source.get_license()
    zarr_root = crate.add_dataset(dest_path='.', properties=properties)

    instrument_properties = {'instrument':
        {
            "@id": "#microscope-001",
            "@type": "IndividualProduct",
            "name": "Zeiss LSM 900",
            "manufacturer": {
                "@id": "https://ror.org"
            },
            "serialNumber": "12345-XYZ"
        }
    }
    instrument_entity = ContextEntity(crate, properties=instrument_properties)

    additional_properties = {
        'additionalProperty': [
        {
            "@id": "#acq:001",
            "@type": "PropertyValue",
            "name": "MeanBeamCharge",
            "value": "1.0"
        }
    ]}
    properties_entity = ContextEntity(crate, properties=additional_properties)

    # TODO: Consider hasDefinedTerm as a better alternative when using a defined ontology?
    # TODO: Can add variableMeasured for output properties

    acquisition_properties = {
        'fbbi_id': {'@id': 'obo:FBbi_00000257'},
    }
    acquisition_entity = ImageAcquisition(crate, identifier="#acquisition-001", properties=acquisition_properties)

    # add to acquisition_properties from source
    crate.add(acquisition_entity)
    crate.add(instrument_entity)
    crate.add(properties_entity)
    zarr_root["resultOf"] = acquisition_entity

#    crate.add(ComputationalWorkflow(crate, workflow_schema_filename))

    crate.write(dest_path)
    return crate
