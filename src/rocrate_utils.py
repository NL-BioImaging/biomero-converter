# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py


from rocrate.model import ComputationalWorkflow

from src.rembi_extension import ImageAcquisition
from src.zarr_extension import ZarrCrate


def create_ro_crate(source, dest_path={}):
    crate = ZarrCrate()

    properties = {}
    properties['name'] = source.get_name()
    #properties["description"] = source.get_description()
    #properties["license"] = source.get_license()
    crate.add_dataset(dest_path='.', properties=properties)

    acquisition_properties = {
        'fbbi_id': {'@id': 'obo:FBbi_00000257'},
        'instrument': {
            "@id": "#microscope-001",
            "@type": "IndividualProduct",
            "name": "Zeiss LSM 900",
            "manufacturer": {
                "@id": "https://ror.org"
            },
            "serialNumber": "12345-XYZ"
        },
        'additionalProperty': [
            {
                "@id": "#acq:001",
                "@type": "PropertyValue",
                "name": "MeanBeamCharge",
                "value": "1.0"
            },
        ]
    }
    # TODO: Consider hasDefinedTerm as a better alternative when using a defined ontology?
    # TODO: Can add variableMeasured for output properties

    # add to acquisition_properties from source
    crate.add(ImageAcquisition(crate, properties=acquisition_properties))

#    crate.add(ComputationalWorkflow(crate, workflow_schema_filename))

    crate.write(dest_path)
    return crate
