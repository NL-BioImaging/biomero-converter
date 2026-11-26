"""An input's metadata, without converting it: to check what is read from a file before converting it.

Three views:

- source: the vendor metadata as read from the file (what imaging-metadata-converter maps)
- model: that metadata mapped onto the imaging metadata model
- spatial: the pixel size and position (µm) and the rotation (degrees) the conversion writes
"""

import yaml

from biomero_converter.helper import create_source
from biomero_converter.util import to_plain_types

VIEWS = ('source', 'model', 'spatial')


def read_metadata(input_filename, views=VIEWS, **source_kwargs):
    """
    Read the metadata of an input, without converting it.

    Args:
        input_filename (str): Path to the input file (or folder).
        views (sequence): Which views to read, of VIEWS.
        **source_kwargs: Which part of the input (e.g. image_uuid, plate_id), as for convert().

    Returns:
        dict: {view: metadata} for each view asked for, as plain data.
    """
    source = create_source(input_filename, **source_kwargs)
    try:
        source.init_metadata()
        readers = {
            'source': source.get_source_metadata,
            'model': source.get_model_metadata,
            'spatial': lambda: {'pixel_size': source.get_pixel_size_um(), 'position': source.get_position_um(),
                                'rotation': source.get_rotation()},
        }
        return {view: to_plain_types(readers[view]()) for view in views}
    finally:
        source.close()


def format_metadata(metadata):
    """Metadata as YAML text, in the order it was read."""
    return yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True, default_flow_style=False)
