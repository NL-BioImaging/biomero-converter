import numpy as np


class WindowScanner:
    """
    Computes quantile-based min/max window for image channels.
    """

    def __init__(self):
        """
        Initialize WindowScanner.
        """
        self.min = {}
        self.max = {}

    def process(self, data, dim_order, min_quantile=0.01, max_quantile=0.99):
        """
        Processes image data to compute min/max quantiles for each channel.

        Args:
            data (ndarray): Image data.
            dim_order (str): Dimension order string.
            min_quantile (float): Lower quantile.
            max_quantile (float): Upper quantile.
        """
        if 'c' in dim_order:
            nc = data.shape[dim_order.index('c')]
        else:
            nc = 1
        for channeli in range(nc):
            if 'c' in dim_order:
                channel_data = data.take(channeli, axis=dim_order.index('c'))
            else:
                channel_data = data
            min1, max1 = np.quantile(channel_data, q=[min_quantile, max_quantile])
            if data.dtype.kind in ['u', 'i']:
                min1, max1 = int(min1), int(max1)
            if channeli not in self.min:
                self.min[channeli] = min1
            else:
                self.min[channeli] = min(min1, self.min[channeli])
            if channeli not in self.max:
                self.max[channeli] = max1
            else:
                self.max[channeli] = max(max1, self.max[channeli])

    def get_window(self):
        """
        Returns the computed min/max window for channels.

        Returns:
            tuple: (min dict, max dict)
        """
        return self.min, self.max
