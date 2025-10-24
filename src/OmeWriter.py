from abc import ABC


class OmeWriter(ABC):
    """
    Abstract base class for OME writers.
    """

    def write(self, filepath, source, verbose=False, **kwargs) -> str:
        """
        Write image data and metadata to output.

        Args:
            filepath (str): Output file path.
            source (ImageSource): Source object.
            verbose (bool): If True, prints progress info.
            **kwargs: Additional options.

        Returns:
            dict: Containing output_path: str or list Output file path(s), and other optional output.
        """
        # Expect to return output path (or filepath)
        raise NotImplementedError("This method should be implemented by subclasses.")
