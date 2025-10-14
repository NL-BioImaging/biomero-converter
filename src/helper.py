import os.path


def create_source(filename, input_format=None, **kwargs):
    """
    Create an image source object based on the input file extension.

    Args:
        filename (str): Path to the input file or Incucyte .icarch file.
        input_format (str, optional): Deprecated - format is auto-detected.
        **kwargs: Source-specific parameters (e.g., plate_id for Incucyte).

    Returns:
        ImageSource: Source object for the input file.

    Raises:
        ValueError: If the file format is unsupported.
    """
    input_ext = os.path.splitext(filename)[1].lower()

    if input_ext == '.db':
        from src.ImageDbSource import ImageDbSource
        source = ImageDbSource(filename)
    elif input_ext == '.icarch':
        # Incucyte archive file - use parent folder for source
        if not os.path.isfile(filename):
            raise ValueError(
                f'Incucyte archive file not found: {filename}'
            )
        archive_folder = os.path.dirname(filename)
        # Verify EssenFiles folder exists
        essen_path = os.path.join(archive_folder, 'EssenFiles')
        if not os.path.isdir(essen_path):
            raise ValueError(
                f'EssenFiles folder not found in: {archive_folder}. '
                f'Expected Incucyte archive structure.'
            )
        from src.IncucyteSource import IncucyteSource
        # Pass kwargs to IncucyteSource (e.g., plate_id)
        source = IncucyteSource(archive_folder, **kwargs)
    elif input_ext == '.isyntax':
        from src.ISyntaxSource import ISyntaxSource
        source = ISyntaxSource(filename)
    elif 'zar' in input_ext:
        from src.OmeZarrSource import OmeZarrSource
        source = OmeZarrSource(filename)
    elif 'tif' in input_ext:
        from src.TiffSource import TiffSource
        source = TiffSource(filename)
    else:
        raise ValueError(
            f'Unsupported input file format: {input_ext}. '
            f'Supported formats: .db, .icarch, .tif, .tiff'
        )
    return source


def get_incucyte_plates(filename):
    """
    Get all available plate IDs from an Incucyte archive.
    
    Args:
        filename (str): Path to the Incucyte archive folder or .icarch file.
        
    Returns:
        list: List of plate IDs (strings) found in the archive.
        
    Raises:
        ValueError: If the path is not a valid Incucyte archive.
    """
    # If it's an .icarch file, use its parent folder
    if os.path.isfile(filename) and filename.lower().endswith('.icarch'):
        archive_folder = os.path.dirname(filename)
    elif os.path.isdir(filename):
        archive_folder = filename
    else:
        raise ValueError(
            f'Invalid Incucyte archive path. Expected folder or .icarch '
            f'file: {filename}'
        )
    
    from src.IncucyteSource import IncucyteSource
    return IncucyteSource.get_available_plates(archive_folder)


def create_incucyte_source(filename, plate_id=None):
    """
    Create an IncucyteSource object for a specific plate.
    
    Args:
        filename (str): Path to the Incucyte archive folder or .icarch file.
        plate_id (str, optional): Specific plate ID to process. If None,
                                 uses the first available plate if multiple
                                 plates exist.
        
    Returns:
        IncucyteSource: Source object for the specified plate.
        
    Raises:
        ValueError: If the path is not a valid Incucyte archive.
    """
    # If it's an .icarch file, use its parent folder
    if os.path.isfile(filename) and filename.lower().endswith('.icarch'):
        archive_folder = os.path.dirname(filename)
    elif os.path.isdir(filename):
        archive_folder = filename
    else:
        raise ValueError(
            f'Invalid Incucyte archive path. Expected folder or .icarch '
            f'file: {filename}'
        )
    
    from src.IncucyteSource import IncucyteSource
    return IncucyteSource(archive_folder, plate_id=plate_id)


def create_writer(output_format, verbose=False):
    """
    Create a writer object and output extension based on the output format.

    Args:
        output_format (str): Output format string.
        verbose (bool): If True, enables verbose output.

    Returns:
        tuple: (writer object, output file extension)

    Raises:
        ValueError: If the output format is unsupported.
    """
    if 'zar' in output_format:
        if '3' in output_format:
            zarr_version = 3
            ome_version = '0.5'
        else:
            zarr_version = 2
            ome_version = '0.4'
        from src.OmeZarrWriter import OmeZarrWriter
        writer = OmeZarrWriter(zarr_version=zarr_version, ome_version=ome_version, verbose=verbose)
        ext = '.ome.zarr'
    elif 'tif' in output_format:
        from src.OmeTiffWriter import OmeTiffWriter
        writer = OmeTiffWriter(verbose=verbose)
        ext = '.ome.tiff'
    else:
        raise ValueError(f'Unsupported output format: {output_format}')
    return writer, ext
