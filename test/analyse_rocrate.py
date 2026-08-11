import json
from pprint import pprint

from src.TreeBuilder import TreeBuilder


def recreate_tree(data):
    tree_builder = TreeBuilder(data)
    return tree_builder.recreate_tree()


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
    #filename = 'C:/Project/slides/tiff/output_jsons/24-079_Region1_r1_c1_260722174141591 ro-crate-metadata.json'

    with open(filename, 'r') as file:
        data = json.load(file)
        root = data['@graph']
    #extract_types(root)
    pprint(recreate_tree(root), indent=2, sort_dicts=False)
