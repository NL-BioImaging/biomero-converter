import json
import logging
import os.path
import shutil

from src.helper import create_source, create_writer
from src.parameters import CONVERSION_ATTEMPTS
from src.util import print_hbytes


def init_logging(log_filename, verbose=False):
    """
    Initialize logging to file and optionally to console.

    Args:
        log_filename (str): Path to the log file.
        verbose (bool): If True, also log to console.
    """
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


def convert(input_filename, output_folder, input_format=None, alt_output_folder=None,
            output_format='omezarr2', show_progress=False, verbose=False):
    attempts = 0
    while True:
        try:
            return _convert(input_filename, output_folder, alt_output_folder=alt_output_folder,
                            output_format=output_format, show_progress=show_progress, verbose=verbose)
        except Exception as e:
            if attempts >= CONVERSION_ATTEMPTS - 1:
                logging.error(e)
                raise Exception(f'Conversion failed after {CONVERSION_ATTEMPTS} attempts: {input_filename}')
        attempts += 1


def _convert(input_filename, output_folder, alt_output_folder=None,
             output_format='omezarr2', show_progress=False, verbose=False):
    """
    Convert an input file to OME format and write to output folder(s).

    Args:
        input_filename (str): Path to the input file.
        output_folder (str): Output folder path.
        alt_output_folder (str, optional): Alternative output folder path.
        output_format (str): Output format string.
        show_progress (bool): If True, print progress.
        verbose (bool): If True, enable verbose logging.

    Returns:
        str: JSON string with conversion result info array.
    """

    logging.info(f'Importing {input_filename}')
    source = create_source(input_filename, input_format=input_format)
    writer, output_ext = create_writer(output_format, verbose=verbose)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    source.init_metadata()
    if verbose:
        print(f'Total data size:    {print_hbytes(source.get_total_data_size())}')
    name = source.get_name()
    output_path = os.path.join(output_folder, name + output_ext)
    full_output_path = writer.write(output_path, source)
    source.close()

    if show_progress:
        print(f'Converting {input_filename} to {output_path}')

    result = {'name': name}
    if isinstance(full_output_path, list):
        full_path = full_output_path[0]
    else:
        full_path = full_output_path
    result['full_path'] = full_path
    message = f"Exported   {result['full_path']}"

    if alt_output_folder:
        if not os.path.exists(alt_output_folder):
            os.makedirs(alt_output_folder)
        alt_output_path = os.path.join(alt_output_folder, name + output_ext)
        if isinstance(full_output_path, list):
            for path in full_output_path:
                alt_output_path = os.path.join(alt_output_folder, os.path.basename(path))
                shutil.copy2(path, alt_output_path)
        elif os.path.isdir(full_output_path):
            shutil.copytree(full_output_path, alt_output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(full_output_path, alt_output_path)
        result['alt_path'] = os.path.join(alt_output_folder, os.path.basename(full_path))
        message += f' and {result["alt_path"]}'

    logging.info(message)
    if show_progress:
        print(message)

    return json.dumps([result])
