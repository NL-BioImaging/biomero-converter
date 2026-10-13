# https://www.geeksforgeeks.org/time-process_time-function-in-python/

import logging
import time


class Timer(object):
    """
    Context manager for timing code execution and logging the elapsed time.
    """

    def __init__(self, title, auto_unit=True, verbose=True):
        """
        Initialize the Timer.

        Args:
            title (str): Description for the timed block.
            auto_unit (bool): Automatically select time unit (seconds/minutes/hours).
            verbose (bool): If True, log the elapsed time.
        """
        self.title = title
        self.auto_unit = auto_unit
        self.verbose = verbose

    def __enter__(self):
        """
        Start timing.
        """
        self.ptime_start = time.process_time()
        self.time_start = time.time()

    def __exit__(self, type, value, traceback):
        """
        Stop timing and log the elapsed time.

        Args:
            type: Exception type, if any.
            value: Exception value, if any.
            traceback: Exception traceback, if any.
        """
        if self.verbose:
            ptime_end = time.process_time()
            time_end = time.time()
            pelapsed = ptime_end - self.ptime_start
            elapsed = time_end - self.time_start
            unit = 'seconds'
            if self.auto_unit and elapsed >= 60:
                pelapsed /= 60
                elapsed /= 60
                unit = 'minutes'
                if elapsed >= 60:
                    pelapsed /= 60
                    elapsed /= 60
                    unit = 'hours'
            logging.info(f'Time {self.title}: {elapsed:.1f} ({pelapsed:.1f}) {unit}')
