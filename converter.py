import json
import logging
import os.path
import shutil

from src.helper import create_source, create_writer


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

    source.init_metadata()
    name = source.get_name()
    output_path = os.path.join(output_folder, name + output_ext)
    full_output_path = writer.write(output_path, source)
    source.close()

    if show_progress:
        print(f'Converting {input_filename} to {output_path}')

    result = {'name': name}
    if isinstance(full_output_path, list):
        result['full_path'] = full_output_path[0]
    else:
        result['full_path'] = full_output_path
    message = f"Exported   {result['full_path']}"

    if alt_output_folder:
        if not os.path.exists(alt_output_folder):
            os.makedirs(alt_output_folder)
        alt_output_path = os.path.join(alt_output_folder, name + output_ext)
        if isinstance(full_output_path, list):
            for path in full_output_path:
                alt_output_path = os.path.join(alt_output_folder, os.path.basename(path))
                shutil.copy2(path, alt_output_path)
        elif os.path.isdir(output_format):
            shutil.copytree(full_output_path, alt_output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(full_output_path, alt_output_path)
        result['alt_path'] = os.path.join(alt_output_folder, os.path.basename(full_output_path))
        message += f' and {result["alt_path"]}'

    logging.info(message)
    if show_progress:
        print(message)

    return json.dumps([result])
