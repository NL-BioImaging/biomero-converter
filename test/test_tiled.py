"""TiledSource (whole-slide images read by region): background fill and skipped empty chunks, synthetic data."""
import numpy as np
import pytest
import zarr

from biomero_converter.OmeZarrSource import OmeZarrSource
from biomero_converter.OmeZarrWriter import OmeZarrWriter
from biomero_converter.TiledSource import TiledSource


class SlideSource(TiledSource):
    """A slide of white background (transparent) with one scanned square of tissue."""

    def __init__(self, uri='slide', metadata={}, **kwargs):
        super().__init__(uri, metadata, **kwargs)
        self.size, self.scanned = (300, 400), (slice(64, 128), slice(128, 256))   # (y, x)
        self.tissue = np.random.default_rng(0).integers(0, 200, (64, 128, 3), dtype=np.uint8)

    def init_metadata(self):
        height, width = self.size
        self.widths, self.heights = [width], [height]
        self.shape = (height, width, 3)
        self.shapes, self.scales = [self.shape], [1]
        self.dtype, self.nchannels, self.dim_order = np.dtype(np.uint8), 3, 'yxc'
        self.background = (255, 255, 255)
        self.name, self.pixel_size, self.bits_per_pixel, self.acquisition_datetime = 'slide', {'x': 1, 'y': 1}, 8, None
        self.metadata = {}
        y, x = self.scanned
        self.bounds = (x.start, y.start, x.stop - x.start, y.stop - y.start)
        self.crop()
        return self.metadata

    def read_rgba(self, x, y, width, height, level=0):
        rgba = np.zeros(self.size + (4,), dtype=np.uint8)
        rgba[self.scanned + (slice(0, 3),)] = self.tissue
        rgba[self.scanned + (3,)] = 255
        return rgba[y:y + height, x:x + width]

    def get_position_um(self, well_id=None):
        return self.get_crop_position_um()

    def expected(self):
        if self.crop_to_scan:
            return self.tissue
        data = np.full(self.shape, 255, dtype=np.uint8)
        data[self.scanned] = self.tissue
        return data


@pytest.mark.parametrize('ome_version', ['0.4', '0.5'])
def test_background_chunks_not_written(tmp_path, monkeypatch, ome_version):
    import biomero_converter.OmeZarrWriter as writer_module
    import biomero_converter.TiledSource as tiled_module
    monkeypatch.setattr(writer_module, 'ZARR_CHUNK_SIZE', 64)
    monkeypatch.setattr(tiled_module, 'TILE_SIZE', 64)
    monkeypatch.setattr(writer_module, 'fits_in_memory', lambda *args, **kwargs: False)   # lazily, by tile
    source = SlideSource(crop_to_scan=False)   # the whole slide, with its background
    source.init_metadata()
    path = str(tmp_path / 'slide.ome.zarr')
    OmeZarrWriter(ome_version=ome_version).write(path, source, write_xml=False, write_rocrate=False)

    array = zarr.open_group(path, mode='r')['s0']   # full resolution
    assert array.fill_value == 255
    if ome_version == '0.4':
        # only the chunks with tissue are stored: 1 x 2 chunks of 64 x 64, per channel
        # (0.5: shards, which count all their chunks as stored, but leave out empty ones)
        assert array.nchunks_initialized == 3 * 2
    target = OmeZarrSource(path)
    target.init_metadata()
    np.testing.assert_array_equal(np.asarray(target.get_data('yxc')), source.expected())


def test_crop_to_scan(tmp_path, monkeypatch):
    import biomero_converter.OmeZarrWriter as writer_module
    import biomero_converter.TiledSource as tiled_module
    monkeypatch.setattr(writer_module, 'ZARR_CHUNK_SIZE', 64)
    monkeypatch.setattr(tiled_module, 'TILE_SIZE', 48)    # tiles not aligned with the scanned region
    monkeypatch.setattr(writer_module, 'fits_in_memory', lambda *args, **kwargs: False)
    source = SlideSource(crop_to_scan=True)
    source.init_metadata()
    assert source.get_shape() == (64, 128, 3)
    assert source.get_position_um() == {'x': 128, 'y': 64}
    path = str(tmp_path / 'slide.ome.zarr')
    OmeZarrWriter(ome_version='0.4').write(path, source, write_xml=False, write_rocrate=False)

    target = OmeZarrSource(path)
    target.init_metadata()
    np.testing.assert_array_equal(np.asarray(target.get_data('yxc')), source.tissue)


MIRAX_FILE = 'C:/Project/slides/3DHistech/sample4.mrxs'


def low_resolution_mirax(first_level, **kwargs):
    """A MIRAX source starting at a low resolution level: OpenSlide reading and conversion in seconds, not hours."""
    from biomero_converter.MiraxSource import MiraxSource

    class LowResolutionMirax(MiraxSource):
        def init_metadata(self):
            super().init_metadata()
            downsample = self.level_downsamples[first_level]
            self.widths, self.heights = self.widths[first_level:], self.heights[first_level:]
            self.shapes, self.offsets = self.shapes[first_level:], self.offsets[first_level:]
            self.shape = self.shapes[0]
            self.scales = [scale * downsample for scale in self.scales[first_level:]]
            self.pixel_size = {dim: size * downsample for dim, size in self.pixel_size.items()}
            return self.metadata

        def read_rgba(self, x, y, width, height, level=0):
            return super().read_rgba(x, y, width, height, level + first_level)

    return LowResolutionMirax(MIRAX_FILE, **kwargs)


@pytest.mark.integration
@pytest.mark.skipif(not __import__('os').path.exists(MIRAX_FILE), reason='MIRAX test file not available')
@pytest.mark.parametrize('output_format, crop_to_scan',
                         [('omezarr04', False), ('omezarr05', False), ('ometiff', False), ('omezarr04', True)])
def test_mirax_low_resolution_round_trip(tmp_path, monkeypatch, output_format, crop_to_scan):
    import biomero_converter.OmeTiffWriter as tiff_writer_module
    import biomero_converter.OmeZarrWriter as zarr_writer_module
    from biomero_converter.helper import create_source, create_writer
    # lazily, by tile, as a full resolution slide
    monkeypatch.setattr(zarr_writer_module, 'fits_in_memory', lambda *args, **kwargs: False)
    monkeypatch.setattr(tiff_writer_module, 'fits_in_memory', lambda *args, **kwargs: False)
    source = low_resolution_mirax(5, crop_to_scan=crop_to_scan)   # 7184 x 3296 pixels
    source.init_metadata()
    if crop_to_scan:
        full = low_resolution_mirax(5, crop_to_scan=False)
        full.init_metadata()
        (x, y), (height, width) = source.offsets[0], source.get_shape()[:2]
        # no data lost: the cropped image is the scanned part of the slide, and outside it is background only
        full_data = full.get_data('yxc')
        full.close()
        outside = full_data.copy()
        outside[y:y + height, x:x + width] = full.background
        assert np.all(outside == full.background)
    writer, ext = create_writer(output_format)
    path = str(tmp_path / ('slide' + ext))
    try:
        writer.write(path, source, write_xml=False, write_rocrate=False)
        expected = source.get_data('yxc')
    finally:
        source.close()

    target = create_source(path)
    target.init_metadata()
    if crop_to_scan:
        np.testing.assert_array_equal(full_data[y:y + height, x:x + width], expected)
    assert target.get_pixel_size_um()['x'] == pytest.approx(source.get_pixel_size_um()['x'], rel=1e-3)
    np.testing.assert_array_equal(np.asarray(target.get_data('yxc')), expected)
    if 'zarr' in ext:
        assert zarr.open_group(path, mode='r')['s0'].fill_value == source.get_fill_value()
