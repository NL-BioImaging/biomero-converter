import numpy as np

from color_conversion import rgba_to_hexrgb


def create_axes_metadata(dimension_order):
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


def create_transformation_metadata(dimension_order, pixel_size_um, scale, translation_um={}):
    metadata = []
    pixel_size_scale = []
    translation_scale = []
    for dimension in dimension_order:
        pixel_size_scale1 = pixel_size_um.get(dimension, 1)
        if pixel_size_scale1 == 0:
            pixel_size_scale1 = 1
        if dimension in ['x', 'y']:
            pixel_size_scale1 /= scale
        pixel_size_scale.append(pixel_size_scale1)

        translation1 = translation_um.get(dimension, 0)
        if dimension in ['x', 'y']:
            translation1 *= scale
        translation_scale.append(translation1)

    metadata.append({'type': 'scale', 'scale': pixel_size_scale})
    if not all(v == 0 for v in translation_scale):
        metadata.append({'type': 'translation', 'translation': translation_scale})
    return metadata


def create_channel_metadata(dtype, channels, nchannels, ome_version):
    if len(channels) < nchannels:
        labels = []
        colors = []
        if nchannels in (3, 4):
            labels = ['Red', 'Green', 'Blue']
            colors = ["FF0000", "00FF00", "0000FF"]
        if nchannels == 4:
            labels += ['Alpha']
            colors += ["FFFFFF"]
        channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]

    omezarr_channels = []
    for channeli, channel in enumerate(channels):
        omezarr_channel = {'label': channel.get('label', channel.get('Name'), f'{channeli}')}
        color = channel.get('color', channel.get('Color'))
        if color is not None:
            omezarr_channel['color'] = rgba_to_hexrgb(color)
        if dtype.kind == 'f':
            # info = np.finfo(dtype)
            start, end = 0, 1
        else:
            info = np.iinfo(dtype)
            start, end = info.min, info.max
        min, max = start, end
        omezarr_channel['window'] = {'start': start, 'end': end, 'min': min, 'max': max}
        omezarr_channels.append(omezarr_channel)

    metadata = {
        'version': ome_version,
        'channels': omezarr_channels,
    }
    return metadata


def scale_dimensions_xy(shape0, dimension_order, scale):
    shape = []
    if scale == 1:
        return shape0
    for shape1, dimension in zip(shape0, dimension_order):
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape.append(shape1)
    return shape


def scale_dimensions_dict(shape0, scale):
    shape = {}
    if scale == 1:
        return shape0
    for dimension, shape1 in shape0.items():
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape[dimension] = shape1
    return shape
