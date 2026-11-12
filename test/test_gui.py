import gc
import json
import os
import tkinter

import numpy as np
import pytest

customtkinter = pytest.importorskip('customtkinter')
tifffile = pytest.importorskip('tifffile')


@pytest.fixture(scope='module')
def tk_root():
    # one Tk application for all tests: initialising Tk repeatedly in one process can fail on Windows
    # (TclError: invalid command name "tcl_findLibrary")
    try:
        root = customtkinter.CTk()
    except tkinter.TclError as error:     # no display (e.g. headless Linux without xvfb)
        pytest.skip(f'no display for the GUI: {error}')
    root.withdraw()
    yield root
    root.destroy()
    _collect_tk_garbage()


def _collect_tk_garbage():
    # Free unreferenced Tk objects (e.g. CTkFont) now, on the main thread. Otherwise a later garbage collection
    # can run their __del__ in a converter (dask) worker thread: its Tcl call waits for the main thread, which
    # waits for the conversion - a deadlock that hangs the next test that converts.
    gc.collect()


@pytest.fixture
def app(tk_root):
    yield tk_root
    for child in tk_root.winfo_children():     # remove this test's frames and windows
        child.destroy()
    _collect_tk_garbage()


# Deterministic, without running the Tk event loop (update() hangs on headless macOS CI):
# wait for the background threads, then let the GUI handle their results as its timers would.

def wait_until_read(frame, timeout=60):
    for thread in frame.scan_threads:
        thread.join(timeout)
    frame._poll_scans()


def wait_until_done(frame, timeout=180):
    frame.worker.join(timeout)
    assert not frame.busy, 'conversion did not finish in time'
    frame._poll_messages()


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
    from biomero_converter.gui.app import WHOLE_FILE
    input_folder = tmp_path / 'input'
    input_folder.mkdir()
    good = input_folder / 'good.tif'
    tifffile.imwrite(good, np.zeros((32, 32), dtype=np.uint8))
    bad = input_folder / 'bad.tif'
    bad.write_text('not a tiff')

    frame = ConverterFrame(app)
    frame.add_inputs([str(bad), str(good)])
    wait_until_read(frame)
    row = frame.rows[os.path.normpath(str(good))]
    assert row.menu.get() == WHOLE_FILE                               # a TIFF has no parts to choose from
    assert row.menu.cget('state') == 'disabled'
    frame.set_output_folder(str(tmp_path / 'output'))
    frame.format_menu.set('OME-Zarr v0.4')
    frame.start_conversion()
    assert frame.busy
    assert frame.convert_button.cget('state') == 'disabled'
    wait_until_done(frame)

    text = log_text(frame)
    assert 'FAILED' in text                                           # the bad input is reported...
    assert 'Finished: 1/2 converted successfully.' in text            # ...and the next one still converts
    output = tmp_path / 'output' / 'good.ome.zarr'
    assert (output / '.zattrs').exists()                              # OME-Zarr 0.4 (Zarr v2)
    assert frame.progress.get() == 1
    assert frame.convert_button.cget('state') == 'normal'


def test_gui_selection_dropdown(app, tmp_path, monkeypatch):
    # an input with parts: the dropdown lists them, and the choice decides what is converted
    from biomero_converter.gui import ConverterFrame
    from biomero_converter.gui import app as gui_app
    parts = [('First [aaaaaaaa]', {'image_uuid': 'aaaaaaaa-1'}), ('Second [#1]', {'image_index': 1})]
    monkeypatch.setattr(gui_app, 'get_selections', lambda path: parts if path.endswith('.lif') else [])
    calls = []

    def fake_convert(input_path, output_folder, output_format=None, progress=None, **source_kwargs):
        calls.append(source_kwargs)
        return json.dumps([{'name': 'out', 'full_path': os.path.join(output_folder, 'out.ome.zarr')}])

    monkeypatch.setattr(gui_app, 'convert', fake_convert)
    input_path = str(tmp_path / 'experiment.lif')
    frame = ConverterFrame(app)
    frame.add_inputs([input_path])
    wait_until_read(frame)
    row = frame.rows[os.path.normpath(input_path)]
    assert row.menu.cget('values') == ['All images (2)', 'First [aaaaaaaa]', 'Second [#1]']
    assert row.menu.get() == 'All images (2)'
    assert row.menu.cget('state') == 'normal'
    frame.set_output_folder(str(tmp_path / 'output'))

    frame.start_conversion()                    # "All": each part
    wait_until_done(frame)
    assert calls == [{'image_uuid': 'aaaaaaaa-1'}, {'image_index': 1}]
    assert 'Converting' in log_text(frame) and '(Second [#1])' in log_text(frame)

    calls.clear()
    row.menu.set('Second [#1]')                 # one part
    frame.start_conversion()
    assert row.menu.cget('state') == 'disabled'  # no changes while converting
    wait_until_done(frame)
    assert calls == [{'image_index': 1}]
    assert row.menu.cget('state') == 'normal'


def test_gui_window_closes_after_current_conversion(app, tmp_path):
    from biomero_converter.gui import ConverterWindow
    images = []
    for name in ('first.tif', 'second.tif'):
        images.append(str(tmp_path / name))
        tifffile.imwrite(images[-1], np.zeros((32, 32), dtype=np.uint8))
    window = ConverterWindow(app)
    frame = window.frame
    frame.add_inputs(images)
    wait_until_read(frame)
    frame.set_output_folder(str(tmp_path / 'output'))
    frame.start_conversion()
    frame.close()                   # while converting: stop after the current conversion, then close
    assert window.winfo_exists()
    wait_until_done(frame)
    assert not window.winfo_exists()
    # the stop can come before the first conversion starts: at most that one ran, never the second
    output_folder = tmp_path / 'output'
    assert len(os.listdir(output_folder)) <= 1 if output_folder.exists() else True


def test_resolve_output_folder(tmp_path):
    from biomero_converter.gui.app import resolve_output_folder
    data = tmp_path / 'data'
    absolute = str(tmp_path / 'converted')
    assert resolve_output_folder(absolute, str(data / 'a.lif')) == os.path.normpath(absolute)
    assert resolve_output_folder('converted', str(data / 'a.lif')) == os.path.normpath(str(data / 'converted'))
    assert resolve_output_folder('.', str(data / 'a.lif')) == os.path.normpath(str(data))
    # a folder input (e.g. DICOM or OME-Zarr): next to it, not inside it
    assert resolve_output_folder('converted', str(data / 'scan.ome.zarr')) == \
        os.path.normpath(str(data / 'converted'))
    assert resolve_output_folder('converted', str(data / 'dicom_folder') + os.sep) == \
        os.path.normpath(str(data / 'converted'))


def test_gui_relative_output_folder(app, tmp_path, monkeypatch):
    # a relative output path is used next to each input
    from biomero_converter.gui import ConverterFrame
    from biomero_converter.gui import app as gui_app
    monkeypatch.setattr(gui_app, 'get_selections', lambda path: [])
    output_folders = []

    def fake_convert(input_path, output_folder, output_format=None, progress=None, **source_kwargs):
        output_folders.append(output_folder)
        return json.dumps([{'name': 'out', 'full_path': os.path.join(output_folder, 'out.ome.zarr')}])

    monkeypatch.setattr(gui_app, 'convert', fake_convert)
    inputs = [str(tmp_path / 'first' / 'a.tif'), str(tmp_path / 'second' / 'b.tif')]
    frame = ConverterFrame(app)
    frame.add_inputs(inputs)
    wait_until_read(frame)
    frame.set_output_folder('converted')
    frame.start_conversion()
    wait_until_done(frame)
    assert output_folders == [os.path.normpath(str(tmp_path / 'first' / 'converted')),
                              os.path.normpath(str(tmp_path / 'second' / 'converted'))]


def test_gui_progress_bar(app, tmp_path, monkeypatch):
    # one bar for all conversions, including the progress within each conversion
    from biomero_converter.gui import ConverterFrame
    from biomero_converter.gui import app as gui_app
    monkeypatch.setattr(gui_app, 'get_selections', lambda path: [])

    def fake_convert(input_path, output_folder, output_format=None, progress=None, **source_kwargs):
        progress(0.5)
        progress(1.0)
        return json.dumps([{'name': 'out', 'full_path': os.path.join(output_folder, 'out.ome.zarr')}])

    monkeypatch.setattr(gui_app, 'convert', fake_convert)
    frame = ConverterFrame(app)
    frame.add_inputs([str(tmp_path / 'a.tif'), str(tmp_path / 'b.tif')])
    wait_until_read(frame)
    frame.set_output_folder(str(tmp_path / 'output'))
    frame.start_conversion()
    assert frame.status_label.cget('text') == 'Starting...'
    frame.worker.join(60)
    messages = list(frame.messages.queue)
    bar = [content[0] for kind, content in messages if kind == 'progress']
    assert bar == [0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0]     # (index + fraction) / number of conversions
    assert ('progress', (0.75, '2/2: 50%')) in messages
    frame._poll_messages()
    assert frame.progress.get() == 1
    assert frame.status_label.cget('text') == 'Finished'
