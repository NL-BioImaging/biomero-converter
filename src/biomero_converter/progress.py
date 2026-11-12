import time
from contextlib import contextmanager

from dask.callbacks import Callback

# minimum time between reports to the callback (fine-grained progress, e.g. per dask task or tile, is frequent)
REPORT_INTERVAL = 0.1


class Progress:
    """
    Reports the progress of a conversion to an optional callback, as a fraction from 0 to 1.

    Work is divided into steps, each a part of the whole (``step(start, end)``); within a step, finished dask tasks
    and counted items (``items()``) move the progress forward. Progress never goes back. Without a callback
    (e.g. on the command line) nothing is tracked.
    """

    def __init__(self, callback=None, start=0.0, end=1.0, parent=None):
        self.callback = callback
        self.start = start
        self.end = end
        self.parent = parent
        self.fraction = 0.0
        self.last_report_time = 0.0

    @property
    def active(self):
        return self.callback is not None or (self.parent is not None and self.parent.active)

    def report(self, fraction, force=False):
        """Report progress of this part, as a fraction (0 to 1) of it."""
        if not self.active:
            return
        fraction = float(min(max(fraction, self.fraction), 1.0))
        self.fraction = fraction
        if self.parent is not None:
            self.parent.report(self.start + fraction * (self.end - self.start), force=force)
            return
        now = time.monotonic()
        if force or fraction >= 1 or now - self.last_report_time >= REPORT_INTERVAL:
            self.last_report_time = now
            self.callback(fraction)

    def part(self, start, end):
        """A part of the work (from start to end, as fractions of this one), reporting into this one."""
        return Progress(start=start, end=end, parent=self)

    @contextmanager
    def step(self, start, end):
        """A step from start to end (fractions); dask computations during it move the progress within it."""
        part = self.part(start, end)
        if not self.active:
            yield part
            return
        with _DaskProgress(part):
            yield part
        part.report(1, force=True)

    def items(self, iterable, count, start=0.0, end=1.0):
        """Iterate, moving the progress from start to end as the items (expected: count) are taken."""
        for index, item in enumerate(iterable):
            yield item
            if count:
                self.report(start + (end - start) * min((index + 1) / count, 1))


class _DaskProgress(Callback):
    # moves a Progress part forward with the finished tasks of dask computations
    def __init__(self, progress):
        super().__init__()
        self.progress = progress
        self.state = None

    def _start_state(self, dsk, state):
        self.state = state

    def _posttask(self, key, result, dsk, state, worker_id):
        ndone = len(state['finished'])
        nremaining = sum(len(state[name]) for name in ('ready', 'waiting', 'running'))
        if ndone + nremaining:
            self.progress.report(ndone / (ndone + nremaining))
