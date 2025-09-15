import os.path


def create_source(filename, input_format=None):
    """
    Create an image source object based on the input file extension.

    Args:
        filename (str): Path to the input file.

    Returns:
        ImageSource: Source object for the input file.

    Raises:
        ValueError: If the file format is unsupported.
    """
    input_ext = os.path.splitext(filename)[1].lower()

    if input_ext == '.db':
        from src.ImageDbSource import ImageDbSource
        source = ImageDbSource(filename)
    elif 'tif' in input_ext:
        from src.TiffSource import TiffSource
        source = TiffSource(filename)
    elif input_format == 'incucyte':
        #check if filename is a folder
        if not os.path.isdir(filename):
            raise ValueError(f'For Incucyte format, the input should be a folder containing the images, not a file: {filename}')
        else:
            from src.IncucyteSource import IncucyteSource
            source = IncucyteSource(filename)
    else:
        raise ValueError(f'Unsupported input file format: {input_ext}')
    return source

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
