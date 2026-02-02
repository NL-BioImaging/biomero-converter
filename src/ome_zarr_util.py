import numpy as np

from src.color_conversion import rgba_to_hexrgb


def create_axes_metadata(dimension_order):
    """
    Create axes metadata for OME-Zarr from dimension order.

    Args:
        dimension_order (str): String of dimension characters.

    Returns:
        list: List of axis metadata dictionaries.
    """
    axes = []
    for dimension in dimension_order:
        unit1 = None
        if dimension == 't':
            type1 = 'time'
            unit1 = 'millisecond'
        elif dimension == 'c':
            type1 = 'channel'
        else:
            type1 = 'space'
            unit1 = 'micrometer'
        axis = {'name': dimension, 'type': type1}
        if unit1 is not None and unit1 != '':
            axis['unit'] = unit1
        axes.append(axis)
    return axes


def create_transformation_metadata(dimension_order, pixel_size_um, factor, translation_um=None):
    """
    Create transformation metadata (scale and translation) for OME-Zarr.

    Args:
        dimension_order (str): String of dimension characters.
        pixel_size_um (dict): Pixel size in micrometers per dimension.
        factor (float): Scaling factor.
        translation_um (dict, optional): Translation in micrometers per dimension.

    Returns:
        list: List of transformation metadata dictionaries.
    """
    metadata = []
    pixel_size_scale = []
    translation_scale = []
    for dim in dimension_order:
        pixel_size_scale1 = pixel_size_um.get(dim, 1)
        if dim in 'xy':
            pixel_size_scale1 *= factor
        if pixel_size_scale1 == 0:
            pixel_size_scale1 = 1
        pixel_size_scale.append(pixel_size_scale1)

        if translation_um is not None:
            translation1 = translation_um.get(dim, 0)
            # translation_pyramid = translation + (scale - 1) * pixel_size / 2
            if dim in 'xy':
                translation1 += (factor - 1) * pixel_size_um[dim] / 2
            translation_scale.append(translation1)

    metadata.append({'type': 'scale', 'scale': pixel_size_scale})
    if translation_um is not None:
        metadata.append({'type': 'translation', 'translation': translation_scale})
    return metadata


def create_channel_metadata(dtype, channels, nchannels, is_rgb, window, ome_version):
    """
    Create channel metadata for OME-Zarr.

    Args:
        dtype: Numpy dtype of image data.
        channels (list): List of channel dicts.
        nchannels (int): Number of channels.
        window (tuple): Min/max window values.
        ome_version (str): OME-Zarr version.

    Returns:
        dict: Channel metadata dictionary.
    """
    if len(channels) < nchannels:
        labels = []
        colors = []
        if is_rgb and nchannels in (3, 4):
            labels = ['Red', 'Green', 'Blue']
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        if is_rgb and nchannels == 4:
            labels += ['Alpha']
            colors += [(1, 1, 1)]
        channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]

    omezarr_channels = []
    starts, ends = window
    for channeli, channel in enumerate(channels):
        omezarr_channel = {'label': channel.get('label', channel.get('Name', f'{channeli}')), 'active': True}
        color = channel.get('color', channel.get('Color'))
        if color is not None:
            omezarr_channel['color'] = rgba_to_hexrgb(color)
        if np.dtype(dtype).kind == 'f':
            min, max = 0, 1
        else:
            info = np.iinfo(dtype)
            min, max = info.min, info.max
        if starts and ends:
            start, end = starts[channeli], ends[channeli]
        else:
            start, end = min, max
        omezarr_channel['window'] = {'min': min, 'max': max, 'start': start, 'end': end}
        omezarr_channels.append(omezarr_channel)

    metadata = {
        'version': ome_version,
        'channels': omezarr_channels,
    }
    return metadata


def scale_dimensions_xy(shape0, dimension_order, scale):
    """
    Scale x and y dimensions in a shape tuple.

    Args:
        shape0 (tuple): Original shape.
        dimension_order (str): String of dimension characters.
        scale (float): Scaling factor.

    Returns:
        list: Scaled shape.
    """
    shape = []
    if scale == 1:
        return shape0
    for shape1, dimension in zip(shape0, dimension_order):
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape.append(shape1)
    return shape


def scale_dimensions_dict(shape0, scale):
    """
    Scale x and y dimensions in a shape dictionary.

    Args:
        shape0 (dict): Original shape dictionary.
        scale (float): Scaling factor.

    Returns:
        dict: Scaled shape dictionary.
    """
    shape = {}
    if scale == 1:
        return shape0
    for dimension, shape1 in shape0.items():
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape[dimension] = shape1
    return shape
