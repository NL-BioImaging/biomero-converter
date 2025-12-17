import json
import logging
import numpy as np
import os.path
import shutil

from src.helper import create_source, create_writer
from src.parameters import RETRY_ATTEMPTS
from src.util import validate_filename


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


def convert(input_filename, output_folder, alt_output_folder=None,
            output_format='omezarr2', show_progress=False, verbose=False, max_attempts=RETRY_ATTEMPTS, **kwargs):
    attempts = 0
    while True:
        try:
            return _convert(input_filename, output_folder, alt_output_folder=alt_output_folder,
                            output_format=output_format, show_progress=show_progress, verbose=verbose,
                            **kwargs)
        except Exception as e:
            if attempts >= max_attempts - 1:
                logging.error(e)
                raise Exception(f'Conversion failed after {RETRY_ATTEMPTS} attempts: {input_filename}')
        attempts += 1


def _convert(input_filename, output_folder, alt_output_folder=None,
             output_format='omezarr2', show_progress=False, verbose=False, **kwargs):
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
    
    # Check if input is an Incucyte archive and handle multiple plates
    input_ext = os.path.splitext(input_filename)[1].lower()
    if input_ext == '.icarch':
        from src.helper import get_incucyte_plates

        available_plates = get_incucyte_plates(input_filename)

        # If plate_id not specified, process all plates
        if 'plate_id' not in kwargs or kwargs['plate_id'] is None:
            plate_list = '", "'.join(available_plates)
            logging.info(f'Processing {len(available_plates)} '
                         f'plate(s): "{plate_list}"')
            results = []
            for plate_id in available_plates:
                logging.info(f'Processing plate {plate_id}')
                plate_kwargs = kwargs.copy()
                plate_kwargs['plate_id'] = plate_id
                result = _convert_single(
                    input_filename, output_folder,
                    alt_output_folder=alt_output_folder,
                    output_format=output_format,
                    show_progress=show_progress,
                    verbose=verbose,
                    **plate_kwargs)
                results.extend(json.loads(result))
            return json.dumps(results)

    # Single source conversion
    return _convert_single(
        input_filename, output_folder,
        alt_output_folder=alt_output_folder,
        output_format=output_format,
        show_progress=show_progress,
        verbose=verbose,
        **kwargs)


def _convert_single(input_filename, output_folder, alt_output_folder=None,
                    output_format='omezarr2', show_progress=False,
                    verbose=False, **kwargs):
    """
    Convert a single source to OME format.

    Args:
        input_filename (str): Path to the input file.
        output_folder (str): Output folder path.
        alt_output_folder (str, optional): Alternative output folder path.
        output_format (str): Output format string.
        show_progress (bool): If True, print progress.
        verbose (bool): If True, enable verbose logging.
        **kwargs: Source-specific parameters (e.g., plate_id for Incucyte).

    Returns:
        str: JSON string with conversion result info array.
    """

    logging.info(f'Importing {input_filename}')
    source = create_source(input_filename, **kwargs)
    writer, output_ext = create_writer(output_format, verbose=verbose)
    
    source.init_metadata()
    name = source.get_name()

    # For Incucyte sources with plates, organize output in subfolders
    if os.path.splitext(input_filename)[1].lower() == '.icarch' and source.plate_id:
        # Create plate-specific subfolder
        plate_folder = f"plate_{source.plate_id}"
        output_folder = os.path.join(output_folder, plate_folder)
        if alt_output_folder:
            alt_output_folder = os.path.join(alt_output_folder, plate_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, validate_filename(name) + output_ext)
    
    output = writer.write(output_path, source, **kwargs)
    source.close()

    if show_progress:
        print(f'Converting {input_filename} to {output_path}')

    result = {'name': name}
    full_output_path = output['output_path']
    if isinstance(full_output_path, list):
        full_path = full_output_path[0]
    else:
        full_path = full_output_path
    result['full_path'] = full_path
    message = f'Exported   {full_path}'

    if alt_output_folder:
        if not os.path.exists(alt_output_folder):
            os.makedirs(alt_output_folder)

        alt_output_path = os.path.join(alt_output_folder, os.path.basename(full_path))
        if isinstance(full_output_path, list):
            for path in full_output_path:
                alt_path = os.path.join(alt_output_folder, os.path.basename(path))
                shutil.copy2(path, alt_path)
        elif os.path.isdir(full_output_path):
            shutil.copytree(full_output_path, alt_output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(full_output_path, alt_output_path)

        result['alt_path'] = alt_output_path
        message += f' and {alt_output_path}'

    if 'window' in output:
        window = np.array(output['window']).tolist()
        result['keyvalues'] = [{"channel_mins": window[0], "channel_maxs": window[1]}]

    logging.info(message)
    if show_progress:
        print(message)

    return json.dumps([result])
