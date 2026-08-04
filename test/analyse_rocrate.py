import json
from pprint import pprint


def recreate_tree(data):
    # TODO: read all root items, then find referred ids, and pop them from root
    
    if isinstance(data, dict):
        tree = {key: recreate_tree(value) for key, value in data.items()}
    elif isinstance(data, list):
        tree = [recreate_tree(item) for item in data]
    else:
        tree = data
    return tree


def extract_types(data):
    seen_types = []
    for item in data:
        item_type = item.get('@type')
        if item_type and item_type not in seen_types:
            seen_types.append(item_type)
            pprint(item)


if __name__ == '__main__':
    root_dir = 'C:/Project/AMC/TDCC/Metadata examples/'
    filename = root_dir + 'ARC450/arc-ro-crate-metadata.json'
    with open(filename, 'r') as file:
        data = json.load(file)
        root = data['@graph']
    #extract_types(root)
    pprint(recreate_tree(root))
