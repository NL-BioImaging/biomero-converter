from abc import ABC


class OmeWriter(ABC):
    def write(self, filepath, source, verbose=False, **kwargs) -> str:
        # Expect to return output path (or filepath)
        raise NotImplementedError("This method should be implemented by subclasses.")
