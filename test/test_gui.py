import os
import time

import numpy as np
import pytest

customtkinter = pytest.importorskip('customtkinter')
tifffile = pytest.importorskip('tifffile')


@pytest.fixture
def app():
    try:
        app = customtkinter.CTk()
    except Exception as error:     # no display (e.g. headless Linux without xvfb)
        pytest.skip(f'no display for the GUI: {error}')
    app.withdraw()
    yield app
    if app.winfo_exists():
        app.destroy()


def wait_until_done(app, frame, timeout=180):
    start = time.monotonic()
    while frame.busy or not frame.messages.empty():
        assert time.monotonic() - start < timeout, 'conversion did not finish in time'
        app.update()
        time.sleep(0.05)
    for _ in range(5):              # let the last poll update the widgets
        app.update()
        time.sleep(0.05)


def log_text(frame):
    return frame.log_box.get('0.0', 'end')


def test_gui_defaults(app):
    from biomero_converter.gui import ConverterFrame
    from biomero_converter.gui.app import OUTPUT_FORMATS
    from biomero_converter.parameters import DEFAULT_OUTPUT_FORMAT
    frame = ConverterFrame(app)
    assert OUTPUT_FORMATS[frame.format_menu.get()] == DEFAULT_OUTPUT_FORMAT
    assert frame.convert_button.cget('state') == 'normal'
    frame.start_conversion()        # nothing selected: asks for inputs, does not start
    assert not frame.busy
    assert 'Please add at least one input' in log_text(frame)


def test_gui_converts_and_continues_after_failure(app, tmp_path):
    from biomero_converter.gui import ConverterFrame
    input_folder = tmp_path / 'input'
    input_folder.mkdir()
    good = input_folder / 'good.tif'
    tifffile.imwrite(good, np.zeros((32, 32), dtype=np.uint8))
    bad = input_folder / 'bad.tif'
    bad.write_text('not a tiff')

    frame = ConverterFrame(app)
    frame.add_inputs([str(bad), str(good)])
    frame.set_output_folder(str(tmp_path / 'output'))
    frame.format_menu.set('OME-Zarr v0.4')
    frame.start_conversion()
    assert frame.busy
    assert frame.convert_button.cget('state') == 'disabled'
    wait_until_done(app, frame)

    text = log_text(frame)
    assert 'FAILED' in text                                           # the bad input is reported...
    assert 'Finished: 1/2 input(s) converted successfully.' in text   # ...and the next one still converts
    output = tmp_path / 'output' / 'good.ome.zarr'
    assert (output / '.zattrs').exists()                              # OME-Zarr 0.4 (Zarr v2)
    assert frame.progress.get() == 1
    assert frame.convert_button.cget('state') == 'normal'


def test_gui_window_closes_after_current_input(app, tmp_path):
    from biomero_converter.gui import ConverterWindow
    image = tmp_path / 'image.tif'
    tifffile.imwrite(image, np.zeros((32, 32), dtype=np.uint8))
    window = ConverterWindow(app)
    frame = window.frame
    frame.add_inputs([str(image), str(image)])
    frame.set_output_folder(str(tmp_path / 'output'))
    frame.start_conversion()
    frame.close()                   # while converting: stop after the current input, then close
    assert window.winfo_exists()
    wait_until_done(app, frame)
    assert not window.winfo_exists()
    assert os.listdir(tmp_path / 'output')
