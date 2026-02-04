import imageio.v3 as iio
import os.path
import re

from src.ImageSource import ImageSource


class GenericSource(ImageSource):
    def __init__(self, uri, **kwargs):
        super().__init__(uri, **kwargs)
        self.format = os.path.splitext(uri)[1].lower().lstrip('.')
        self.metadata = None
        im = None
        try:
            im = iio.imopen(uri, 'r')
            self.metadata = im.metadata()
            self.data_func = im.read
        except OSError as error:
            error = str(error)
            match = re.search(r"plugin='\w+'", error)
            if match:
                parts = match.group().split('=')
                if len(parts) == 2:
                    self.format = parts[1].strip("'").lower()
        except Exception:
            if im:
                if hasattr(im, 'legacy_get_reader'):
                    reader = im.legacy_get_reader()
                    self.format = reader.format.name.lower()
                    self.metadata = reader.get_meta_data()
                    self.data_func = reader.get_data

    def init_metadata(self):
        return self.metadata

    def get_data(self, **kwargs):
        return self.data_func(**kwargs)
