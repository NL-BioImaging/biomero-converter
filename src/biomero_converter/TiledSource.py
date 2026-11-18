from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import dask
import dask.array as da
import numpy as np

from biomero_converter.ImageSource import ImageSource
from biomero_converter.parameters import TILE_SIZE
from biomero_converter.util import redimension_data, get_available_memory


class TiledSource(ImageSource):
    """
    Base class for whole-slide (RGBA) sources read by region, in 'yxc' dimension order.

    Subclasses implement read_rgba() and set the attributes widths, heights, shapes and scales (per level),
    dtype, nchannels (3: RGB) and dim_order ('yxc'); optionally background (RGB, default black) to fill the
    transparent (unscanned) area with.

    With crop_to_scan, a subclass that knows the scanned region sets bounds (x, y, width, height in level 0 pixels)
    and calls crop() at the end of init_metadata(): the levels are then cropped to it, leaving out the unscanned
    (transparent) area around it.

    A reader that can only read on the thread that opened it sets process_reads = True and implements the static
    methods open_reader() and read_rgba_from(): the (multi-threaded) dask reads then go to worker processes, each
    with its own reader, while dask's threads still resize, compress and write the tiles.
    """

    background = (0, 0, 0)
    process_reads = False
    # estimated memory of a worker process (its reader's tile cache), to limit their number by the available memory
    process_memory = 1e9
    _pool = None
    bounds = None       # scanned region (x, y, width, height) in level 0 pixels, if known
    offsets = None      # per level: (x, y) of the (cropped) image in the level

    def __init__(self, uri, metadata={}, crop_to_scan=True, **kwargs):
        """
        Args:
            crop_to_scan (bool): Crop to the scanned region (bounds), if known (default): no data lost, as the area
                outside is transparent (background), but written about twice as fast.
        """
        super().__init__(uri, metadata)
        self.crop_to_scan = crop_to_scan

    def crop(self):
        """
        Crop the levels (widths, heights, shapes) to the scanned region (bounds), with crop_to_scan.
        """
        self.offsets = [(0, 0)] * len(self.shapes)
        if not self.crop_to_scan or self.bounds is None:
            return
        x0, y0, width0, height0 = self.bounds
        self.offsets, self.shapes = [], []
        for scale, width, height in zip(self.scales, self.widths, self.heights):
            # include partial pixels at the edges: covers the whole scanned region at every level
            x, y = int(np.floor(x0 * scale)), int(np.floor(y0 * scale))
            x1 = min(int(np.ceil((x0 + width0) * scale)), width)
            y1 = min(int(np.ceil((y0 + height0) * scale)), height)
            self.offsets.append((x, y))
            self.shapes.append((y1 - y, x1 - x, self.nchannels))
        self.heights = [shape[0] for shape in self.shapes]
        self.widths = [shape[1] for shape in self.shapes]
        self.shape = self.shapes[0]

    def get_crop_position_um(self):
        """
        The position of the (cropped) image in the slide.
        """
        x, y = self.offsets[0] if self.offsets else (0, 0)
        return {'x': x * self.pixel_size['x'], 'y': y * self.pixel_size['y']}

    def read_rgba(self, x, y, width, height, level=0):
        """
        Read a region of a level as an RGBA (y, x, 4) array; x, y, width and height in that level's pixels.
        """
        raise NotImplementedError("The 'read_rgba' method must be implemented by subclasses.")

    @staticmethod
    def open_reader(uri):
        """
        Open a reader for a worker process (process_reads).
        """
        raise NotImplementedError("The 'open_reader' method must be implemented by process_reads subclasses.")

    @staticmethod
    def read_rgba_from(reader, x, y, width, height, level=0):
        """
        Read a region as RGBA with a reader from open_reader() (process_reads), as read_rgba().
        """
        raise NotImplementedError("The 'read_rgba_from' method must be implemented by process_reads subclasses.")

    def read_array(self, x, y, width, height, level=0):
        """
        Read a region of a level as an RGB (y, x, 3) array, with transparent pixels filled with the background.
        """
        return self._rgba_to_rgb(self.read_rgba(x, y, width, height, level))

    def _read_array_in_process(self, x, y, width, height, level=0):
        try:
            rgba = self._pool.submit(_read_rgba_in_worker, x, y, width, height, level).result()
        except BrokenProcessPool as error:
            raise RuntimeError(f'Reading {self.uri} in worker processes failed: out of memory, or a script '
                               f"converting it without \"if __name__ == '__main__':\" (the processes import it)"
                               ) from error
        return self._rgba_to_rgb(rgba)

    def _rgba_to_rgb(self, rgba):
        alpha = rgba[..., 3]
        if not alpha.any():
            # not scanned (most tiles of a slide): background
            return np.broadcast_to(np.asarray(self.background, dtype=np.uint8), rgba.shape[:2] + (3,))
        if alpha.min() == 255:
            return np.ascontiguousarray(rgba[..., :3])
        alpha = np.atleast_3d(alpha / np.float32(255))
        background = np.asarray(self.background, dtype=np.float32)
        return (rgba[..., :3] * alpha + background * (1 - alpha)).astype(np.uint8)

    def get_data_as_dask(self, dim_order, level=0, **kwargs):
        if self.process_reads:
            if self._pool is None:
                # half the CPUs and memory: dask's threads compress and write (shards) meanwhile; more processes read
                # no faster
                max_workers = get_available_memory() / 2 // self.process_memory
                nworkers = int(max(min(dask.system.CPU_COUNT // 2, max_workers), 1))
                # spawn: a forked process would read on a copy of a thread without the reader's per-thread state
                self._pool = ProcessPoolExecutor(nworkers, mp_context=multiprocessing.get_context('spawn'),
                                                 initializer=_init_worker,
                                                 initargs=(type(self).open_reader, type(self).read_rgba_from,
                                                           self.uri))
            read_array = self._read_array_in_process
        else:
            read_array = self.read_array

        x_offset, y_offset = self.offsets[level] if self.offsets else (0, 0)

        def get_lazy_tile(x, y, width, height):
            lazy_array = dask.delayed(read_array)(int(x + x_offset), int(y + y_offset), width, height, level)
            return da.from_delayed(lazy_array, shape=(height, width, self.nchannels), dtype=self.dtype)

        y_chunks, x_chunks = da.core.normalize_chunks(TILE_SIZE, self.shapes[level][:2], dtype=self.dtype)
        y_pos = np.cumsum([0] + list(y_chunks)[:-1])
        x_pos = np.cumsum([0] + list(x_chunks)[:-1])
        data = da.concatenate(
            [da.concatenate([get_lazy_tile(x, y, width, height) for x, width in zip(x_pos, x_chunks)], axis=1)
             for y, height in zip(y_pos, y_chunks)], axis=0)
        return redimension_data(data, self.dim_order, dim_order)

    def get_pixel_size_um(self):
        return self.pixel_size

    def get_fill_value(self):
        # one value for all channels (chunks hold one channel): a grey background, e.g. white
        background = np.asarray(self.background)
        return int(background[0]) if np.all(background == background[0]) else None

    def get_channels(self):
        return [
            {"name": "Red", "color": [1, 0, 0, 1]},
            {"name": "Green", "color": [0, 1, 0, 1]},
            {"name": "Blue", "color": [0, 0, 1, 1]},
        ]

    def is_rgb(self):
        return True

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None


# worker process (process_reads): its own reader, opened (and so read) on the worker's thread
_worker_reader = None
_worker_read = None


def _init_worker(open_reader, read_rgba_from, uri):
    global _worker_reader, _worker_read
    _worker_reader = open_reader(uri)
    _worker_read = read_rgba_from


def _read_rgba_in_worker(x, y, width, height, level):
    return _worker_read(_worker_reader, x, y, width, height, level)
