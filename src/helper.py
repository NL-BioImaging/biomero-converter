import os.path


def create_source(filename):
    input_ext = os.path.splitext(filename)[1].lower()

    if input_ext == '.db':
        from src.ImageDbSource import ImageDbSource
        source = ImageDbSource(filename)
    elif input_ext == '.isyntax':
        from src.ISyntaxSource import ISyntaxSource
        source = ISyntaxSource(filename)
    elif 'tif' in input_ext:
        from src.TiffSource import TiffSource
        source = TiffSource(filename)
    else:
        raise ValueError(f'Unsupported input file format: {input_ext}')
    return source


def create_writer(output_format, verbose=False):
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
