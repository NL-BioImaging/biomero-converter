import numpy as np


class WindowScanner:
    """
    Computes quantile-based min/max window for image channels.
    """

    def __init__(self):
        """
        Initialize WindowScanner.
        """
        self.mins = []
        self.maxs = []

    def process(self, data, dim_order, min_quantile=0.01, max_quantile=0.99):
        """
        Processes image data to compute min/max quantiles for each channel.

        Args:
            data (ndarray): Image data.
            dim_order (str): Dimension order string.
            min_quantile (float): Lower quantile.
            max_quantile (float): Upper quantile.
        """
        axis = []
        if 't' in dim_order:
            axis += [dim_order.index('t')]
        if 'z' in dim_order:
            axis += [dim_order.index('z')]
        axis += [dim_order.index('y'), dim_order.index('x')]
        values = np.quantile(data, axis=axis, q=[min_quantile, max_quantile])
        mins, maxs = values
        if len(self.mins) == 0:
            self.mins = mins
            self.maxs = maxs
        else:
            self.mins = np.min([mins, self.mins], axis=0)
            self.maxs = np.max([maxs, self.maxs], axis=0)

    def get_window(self):
        """
        Returns the computed min/max window for channels.

        Returns:
            tuple: (min dict, max dict)
        """
        return np.array(self.mins).tolist(), np.array(self.maxs).tolist()
