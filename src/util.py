from datetime import datetime, timedelta
import numpy as np
import os.path
import re


def ensure_list(item):
    if not isinstance(item, (list, tuple)):
        item = [item]
    return item


def redimension_data(data, old_order, new_order, **indices):
    # able to provide optional dimension values e.g. t=0, z=0
    if new_order == old_order:
        return data

    new_data = data
    order = old_order
    # remove
    for o in old_order:
        if o not in new_order:
            index = order.index(o)
            dim_value = indices.get(o, 0)
            new_data = np.take(new_data, indices=dim_value, axis=index)
            order = order[:index] + order[index + 1:]
    # add
    for o in new_order:
        if o not in order:
            new_data = np.expand_dims(new_data, 0)
            order = o + order
    # move
    old_indices = [order.index(o) for o in new_order]
    new_indices = list(range(len(new_order)))
    new_data = np.moveaxis(new_data, old_indices, new_indices)
    return new_data


def get_numpy_data(data, dim_order, t, c, z, y, x, y_size, x_size):
    x_index = dim_order.index('x')
    y_index = dim_order.index('y')
    slices = [slice(None)] * len(dim_order)
    if 't' in dim_order:
        slices[dim_order.index('t')] = t
    if 'c' in dim_order:
        slices[dim_order.index('c')] = c
    if 'z' in dim_order:
        slices[dim_order.index('z')] = z
    slices[y_index] = slice(y, y + y_size)
    slices[x_index] = slice(x, x + x_size)
    return data[tuple(slices)]


def get_level_from_scale(source_scales, target_scale=1):
    best_level_scale = 0, target_scale
    for level, scale in enumerate(source_scales):
        if np.isclose(scale, target_scale, rtol=1e-4):
            return level, 1
        if scale <= target_scale:
            best_level_scale = level, target_scale / scale
    return best_level_scale


def validate_filename(filename):
    return re.sub(r'[^\w_.)(-]', '_', filename)


def get_filetitle(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def splitall(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def split_well_name(well_name, remove_leading_zeros=True, col_as_int=False):
    matches = re.findall(r'(\D+)(\d+)', well_name)
    if len(matches) > 0:
        row, col = matches[0]
        if col_as_int or remove_leading_zeros:
            try:
                col = int(col)
            except ValueError:
                pass
        if not col_as_int:
            col = str(col)
        return row, col
    else:
        raise ValueError(f"Invalid well name format: {well_name}. Expected format like 'A1', 'B2', etc.")


def pad_leading_zero(input_string, num_digits=2):
    output = str(input_string)
    is_well = not output[0].isdigit()
    if is_well:
        row, col = split_well_name(output, remove_leading_zeros=True)
        output = str(col)
    while len(output) < num_digits:
        output = '0' + output
    if is_well:
        output = row + output
    return output


def strip_leading_zeros(well_name):
    row, col = split_well_name(well_name, remove_leading_zeros=True)
    return f'{row}{col}'


def get_rows_cols_plate(nwells):
    nrows_cols = {
        6: (2, 3),
        12: (3, 4),
        24: (4, 6),
        48: (6, 8),
        96: (8, 12),
        384: (16, 24)
    }
    nrows, ncols = nrows_cols[nwells]
    rows = [chr(ord('A') + i) for i in range(nrows)]
    cols = [str(i + 1) for i in range(ncols)]
    return rows, cols


def convert_dotnet_ticks_to_datetime(net_ticks):
    return datetime(1, 1, 1) + timedelta(microseconds=net_ticks // 10)


def xml_content_to_dict(element):
    key = element.tag
    children = list(element)
    if key == 'Array':
        res = [xml_content_to_dict(child) for child in children]
        return res
    if len(children) > 0:
        if children[0].tag == 'Array':
            value = []
        else:
            value = {}
        for child in children:
            child_value = xml_content_to_dict(child)
            if isinstance(child_value, list):
                value.extend(child_value)
            else:
                value |= child_value
    else:
        value = element.text
        if value is not None:
            if '"' in value:
                value = value.replace('"', '')
            else:
                for t in (float, int, bool):
                    try:
                        if t == bool:
                            if value.lower() == 'true':
                                value = True
                            if value.lower() == 'false':
                                value = False
                        else:
                            value = t(value)
                        break
                    except (TypeError, ValueError):
                        pass

    if key == 'DataObject':
        key = element.attrib['ObjectType']
    if key == 'Attribute':
        key = element.attrib['Name']
    return {key: value}


def convert_to_um(value, unit):
    conversions = {
        'nm': 1e-3,
        'µm': 1, 'um': 1, 'micrometer': 1, 'micron': 1,
        'mm': 1e3, 'millimeter': 1e3,
        'cm': 1e4, 'centimeter': 1e4,
        'm': 1e6, 'meter': 1e6
    }
    return value * conversions.get(unit, 1)


def print_dict(value, tab=0, max_len=250, bullet=False):
    s = ''
    if isinstance(value, dict):
        for key, subvalue in value.items():
            s += '\n'
            if bullet:
                s += '-'
                bullet = False
            s += '\t' * tab + str(key) + ': '
            if isinstance(subvalue, dict):
                s += print_dict(subvalue, tab+1)
            elif isinstance(subvalue, list):
                for v in subvalue:
                    s += print_dict(v, tab+1, bullet=True)
            else:
                subvalue = str(subvalue)
                if len(subvalue) > max_len:
                    subvalue = subvalue[:max_len] + '...'
                s += subvalue
    else:
        s += str(value) + ' '
    return s


def print_hbytes(nbytes):
    exps = ['', 'K', 'M', 'G', 'T', 'P', 'E']
    div = 1024
    exp = 0
    while nbytes > div:
        nbytes /= div
        exp += 1
    if exp < len(exps):
        e = exps[exp]
    else:
        e = f'e{exp * 3}'
    return f'{nbytes:.1f}{e}B'
