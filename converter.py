import json
import logging
import os.path
import shutil

from src.helper import create_source, create_writer
from src.util import print_dict, print_hbytes


def init_logging(log_filename, verbose=False):
    basepath = os.path.dirname(log_filename)
    if basepath and not os.path.exists(basepath):
        os.makedirs(basepath)
    handlers = [logging.FileHandler(log_filename, encoding='utf-8')]
    if verbose:
        handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=handlers,
                        encoding='utf-8')

    logging.getLogger('ome_zarr').setLevel(logging.WARNING)     # mute verbose ome_zarr logging


def convert(input_filename, output_folder, alt_output_folder=None,
            output_format='omezarr2', show_progress=False, verbose=False):

    logging.info(f'Importing {input_filename}')
    source = create_source(input_filename)
    writer, output_ext = create_writer(output_format, verbose=verbose)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metadata = source.init_metadata()
    if verbose:
        print(print_dict(metadata))
        print()
        if source.is_screen():
            print(source.print_well_matrix())
            print(source.print_timepoint_well_matrix())
        print(f'Total data size:    {print_hbytes(source.get_total_data_size())}')

    name = source.get_name()
    output_path = os.path.join(output_folder, name + output_ext)
    writer.write(output_path, source, name=name)
    source.close()

    if show_progress:
        print(f'Converting {input_filename} to {output_path}')

    message = f'Exported  {output_path}'
    result = {'name': name, 'full_path': output_path}
    if alt_output_folder:
        if not os.path.exists(alt_output_folder):
            os.makedirs(alt_output_folder)
        alt_output_path = os.path.join(alt_output_folder, name + output_ext)
        if 'zar' in output_format:
            shutil.copytree(output_path, alt_output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(output_path, alt_output_path)
        result['alt_path'] = alt_output_path
        message += f' and {alt_output_path}'

    logging.info(message)
    if show_progress:
        print(message)

    return json.dumps([result])
