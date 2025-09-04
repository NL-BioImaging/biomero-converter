from datetime import datetime, timedelta
import os.path
import re


def ensure_list(item):
    if not isinstance(item, (list, tuple)):
        item = [item]
    return item


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
