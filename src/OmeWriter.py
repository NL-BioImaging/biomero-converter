from abc import ABC


class OmeWriter(ABC):
    def write(self, filename, source, verbose=False, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
