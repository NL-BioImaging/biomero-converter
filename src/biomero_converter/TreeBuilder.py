class TreeBuilder:
    def __init__(self, data):
        self.root = data
        self.index_ids()

    def index_ids(self):
        self.id_index = {item['@id']: item for item in self.root if isinstance(item, dict) and '@id' in item}

    def find_id(self, id_value):
        return self.id_index.get(id_value)

    def recreate_tree(self, data=None):
        if data is None:
            data = self.root

        if isinstance(data, dict) and len(data) == 1 and '@id' in data:
            data = self.find_id(data['@id'])

        if isinstance(data, dict):
            tree = {key: self.recreate_tree(value) for key, value in data.items()}
        elif isinstance(data, list):
            tree = [self.recreate_tree(item) for item in data]
        else:
            tree = data
        return tree
