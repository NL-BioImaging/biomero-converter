import json
import logging
import os.path
import queue
import threading
from tkinter import filedialog

import customtkinter

from biomero_converter.converter import convert
from biomero_converter.helper import LEICA_EXTENSIONS, get_selections
from biomero_converter.parameters import DEFAULT_OUTPUT_FORMAT, VERSION

OUTPUT_FORMATS = {
    'OME-Zarr v0.4': 'omezarr04',
    'OME-Zarr v0.5': 'omezarr05',
    'OME-Zarr v0.6': 'omezarr06',
    'OME-TIFF': 'ometiff',
}
DEFAULT_FORMAT_LABEL = next(label for label, output_format in OUTPUT_FORMATS.items()
                            if output_format == DEFAULT_OUTPUT_FORMAT)

# file dialog filters: tuples of patterns work on Windows, macOS and Linux
INPUT_FILETYPES = [
    ('Supported files', ('*.lif', '*.lof', '*.lifext', '*.xlif', '*.xlcf', '*.xlef', '*.icarch', '*.isyntax',
                         '*.mrxs', '*.db', '*.dcm', '*.dicom', '*.tif', '*.tiff', '*.ome', '*.xml')),
    ('Leica', ('*.lif', '*.lof', '*.lifext', '*.xlif', '*.xlcf', '*.xlef')),
    ('Incucyte archive', ('*.icarch',)),
    ('Whole slide (iSyntax, MIRAX)', ('*.isyntax', '*.mrxs')),
    ('Image database', ('*.db',)),
    ('DICOM', ('*.dcm', '*.dicom')),
    ('TIFF / OME-TIFF', ('*.tif', '*.tiff')),
    ('All files', '*'),
]

# selection dropdown entries without parts to choose from
WHOLE_FILE = 'Whole file'
READING = 'Reading contents...'

TITLE = f'biomero-converter {VERSION}'
POLL_INTERVAL_MS = 100
MAX_PATH_CHARS = 60       # longer input paths are shortened from the left, keeping the file name visible


def _short_path(path, max_chars=MAX_PATH_CHARS):
    if len(path) <= max_chars:
        return path
    tail = path[-(max_chars - 1):]
    # start at a folder boundary when possible
    separator = min((index for index in (tail.find(os.sep), tail.find('/')) if index >= 0), default=-1)
    return '…' + (tail[separator:] if separator >= 0 else tail)


def resolve_output_folder(output_folder, input_path):
    """The output folder for an input: an absolute folder as is; a relative path from the folder that contains the
    input (also for folder inputs, so output is not written inside e.g. an OME-Zarr or DICOM folder)."""
    output_folder = os.path.expanduser(output_folder)
    if os.path.isabs(output_folder):
        return os.path.normpath(output_folder)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.normpath(input_path)), output_folder))


def _part_kind(input_path, selections):
    # what the parts of an input are called in the "All ..." entry
    if 'plate_id' in selections[0][1]:
        return 'plates'
    if os.path.splitext(input_path)[1].lower() in LEICA_EXTENSIONS:
        return 'images'
    return 'series'


class _QueueLogHandler(logging.Handler):
    """Forwards log records from the conversion thread to the GUI thread."""

    def __init__(self, messages):
        super().__init__(level=logging.INFO)
        self.messages = messages
        self.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S'))

    def filter(self, record):
        # INFO from biomero-converter itself (it logs to the root logger),
        # only warnings and errors from its dependencies (linkml, ome_zarr, ...)
        return (record.levelno >= logging.WARNING
                or record.name == 'root' or record.name.startswith('biomero_converter'))

    def emit(self, record):
        try:
            self.messages.put(('log', self.format(record)))
        except Exception:
            self.handleError(record)


class _InputRow:
    """One input in the input list: its path, and a dropdown to choose which part of it to convert."""

    def __init__(self, master, row, path):
        self.path = path
        self.selections = []            # (label, source kwargs) per part; empty: the whole file
        self.all_label = None           # dropdown entry that converts every part
        self.label = customtkinter.CTkLabel(master, text=_short_path(path), anchor='w')
        self.label.grid(row=row, column=0, sticky='ew', pady=2, padx=5)
        self.menu = customtkinter.CTkOptionMenu(master, values=[READING], width=300, dynamic_resizing=False)
        self.menu.grid(row=row, column=1, sticky='e', pady=2, padx=5)
        self.menu.configure(state='disabled')
        self.ready = False

    def set_selections(self, selections):
        self.selections = selections
        if selections:
            self.all_label = f'All {_part_kind(self.path, selections)} ({len(selections)})'
            self.menu.configure(values=[self.all_label] + [label for label, _ in selections], state='normal')
            self.menu.set(self.all_label)
        else:
            self.menu.configure(values=[WHOLE_FILE])
            self.menu.set(WHOLE_FILE)
        self.ready = True

    def set_state(self, state):
        # only inputs with parts to choose from have an enabled dropdown
        self.menu.configure(state=state if self.selections else 'disabled')

    def get_parts(self):
        """The (label, source kwargs) to convert for the current choice."""
        choice = self.menu.get()
        if not self.ready or not self.selections:
            return [(None, {})]
        if choice == self.all_label:
            return list(self.selections)
        return [(label, kwargs) for label, kwargs in self.selections if label == choice]

    def destroy(self):
        self.label.destroy()
        self.menu.destroy()


class ConverterFrame(customtkinter.CTkFrame):
    """The converter interface: inputs (with which part of each to convert), output folder, format, log and
    progress.

    The contents of each input (Leica images, Incucyte plates, DICOM series) are read in the background when it
    is added, to fill its dropdown. Conversions run in a background thread, one after the other; a failed
    conversion is logged and the others continue.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.rows = {}                      # input path -> _InputRow
        self.messages = queue.Queue()       # conversion thread -> GUI thread
        self.worker = None
        self.stop_requested = threading.Event()
        self.close_when_done = False
        self.scan_results = queue.Queue()   # content reading threads -> GUI thread
        self.scan_threads = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

        customtkinter.CTkLabel(
            self,
            text='Convert microscopy files to OME-Zarr or OME-TIFF.\n'
                 'Supports Leica, Incucyte, iSyntax, MIRAX, DICOM, image .db, (OME-)TIFF and OME-Zarr.',
            anchor='center').grid(row=0, column=0, sticky='new', pady=5, padx=10)

        # inputs, each with a dropdown to choose what to convert
        input_frame = customtkinter.CTkFrame(self)
        input_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=10)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_rowconfigure(1, weight=1)
        customtkinter.CTkLabel(input_frame, text='Input files / folders', anchor='w').grid(
            row=0, column=0, sticky='w', pady=(5, 0), padx=5)
        customtkinter.CTkLabel(input_frame, text='Convert (image / plate / series)', anchor='e').grid(
            row=0, column=0, sticky='e', pady=(5, 0), padx=5)
        self.input_list = customtkinter.CTkScrollableFrame(input_frame, height=120)
        self.input_list.grid(row=1, column=0, rowspan=4, sticky='nsew', pady=5, padx=5)
        self.input_list.grid_columnconfigure(0, weight=1)
        self.add_files_button = customtkinter.CTkButton(input_frame, text='Add files', command=self.browse_files)
        self.add_files_button.grid(row=1, column=1, sticky='ew', pady=5, padx=5)
        self.add_folder_button = customtkinter.CTkButton(input_frame, text='Add folder', command=self.browse_folder)
        self.add_folder_button.grid(row=2, column=1, sticky='ew', pady=5, padx=5)
        self.clear_button = customtkinter.CTkButton(input_frame, text='Clear', command=self.clear_inputs)
        self.clear_button.grid(row=3, column=1, sticky='ew', pady=5, padx=5)

        # output folder
        output_frame = customtkinter.CTkFrame(self)
        output_frame.grid(row=2, column=0, sticky='ew', pady=5, padx=10)
        output_frame.grid_columnconfigure(1, weight=1)
        customtkinter.CTkLabel(output_frame, text='Output folder:', anchor='w').grid(
            row=0, column=0, sticky='w', pady=5, padx=5)
        # an absolute folder, or a relative path (e.g. "converted") used next to each input
        self.output_entry = customtkinter.CTkEntry(
            output_frame, placeholder_text='a folder, or a relative path (e.g. converted) next to each input')
        self.output_entry.grid(row=0, column=1, sticky='ew', pady=5, padx=5)
        self.output_button = customtkinter.CTkButton(output_frame, text='Browse', command=self.browse_output)
        self.output_button.grid(row=0, column=2, sticky='ew', pady=5, padx=5)

        # output format
        options = customtkinter.CTkFrame(self)
        options.grid(row=3, column=0, sticky='ew', pady=5, padx=10)
        customtkinter.CTkLabel(options, text='Output format:', anchor='w').grid(
            row=0, column=0, sticky='w', pady=5, padx=5)
        self.format_menu = customtkinter.CTkOptionMenu(options, values=list(OUTPUT_FORMATS))
        self.format_menu.set(DEFAULT_FORMAT_LABEL)
        self.format_menu.grid(row=0, column=1, sticky='w', pady=5, padx=5)

        # log
        self.log_box = customtkinter.CTkTextbox(self, wrap='word')
        self.log_box.grid(row=4, column=0, sticky='nsew', pady=5, padx=10)
        self.log_box.configure(state='disabled')

        # progress, stop and convert
        run_frame = customtkinter.CTkFrame(self, fg_color='transparent')
        run_frame.grid(row=5, column=0, sticky='ew', pady=(5, 10), padx=10)
        run_frame.grid_columnconfigure(0, weight=1)
        self.progress = customtkinter.CTkProgressBar(run_frame)
        self.progress.set(0)
        self.progress.grid(row=0, column=0, sticky='ew', pady=5, padx=5)
        self.stop_button = customtkinter.CTkButton(
            run_frame, text='Stop after current', state='disabled', command=self.request_stop)
        self.stop_button.grid(row=0, column=1, sticky='e', pady=5, padx=5)
        self.convert_button = customtkinter.CTkButton(run_frame, text='Convert', command=self.start_conversion)
        self.convert_button.grid(row=0, column=2, sticky='e', pady=5, padx=5)

    @property
    def inputs(self):
        return list(self.rows)

    @property
    def output_folder(self):
        """As entered: an absolute folder, or a relative path used next to each input (see resolve_output_folder)."""
        return self.output_entry.get().strip()

    @property
    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    # inputs and output

    def browse_files(self):
        self.add_inputs(filedialog.askopenfilenames(parent=self, title='Select input files',
                                                    filetypes=INPUT_FILETYPES))

    def browse_folder(self):
        # folder-based inputs, e.g. OME-Zarr (.zarr directories) or DICOM folders
        path = filedialog.askdirectory(parent=self, title='Select input folder (OME-Zarr, DICOM, ...)')
        if path:
            self.add_inputs([path])

    def add_inputs(self, paths):
        """Add inputs; their contents are read in the background to fill their dropdowns."""
        for path in paths:
            path = os.path.normpath(path)
            if path in self.rows:
                continue
            self.rows[path] = _InputRow(self.input_list, len(self.rows), path)
            thread = threading.Thread(target=self._read_contents, args=(path,), daemon=True)
            self.scan_threads.append(thread)
            thread.start()
        self.after(POLL_INTERVAL_MS, self._poll_scans)

    def _read_contents(self, path):
        # runs in a reading thread: only communicates with the GUI through self.scan_results
        try:
            self.scan_results.put((path, get_selections(path), None))
        except Exception as error:
            self.scan_results.put((path, [], error))

    def _poll_scans(self):
        try:
            while True:
                path, selections, error = self.scan_results.get_nowait()
                row = self.rows.get(path)
                if row is None:         # removed (Clear) while reading
                    continue
                if error is not None:
                    self.log(f'Could not read the contents of {path} ({error}); it is converted as a whole.')
                row.set_selections(selections)
                row.set_state('disabled' if self.busy else 'normal')
        except queue.Empty:
            pass
        self.scan_threads = [thread for thread in self.scan_threads if thread.is_alive()]
        if (self.scan_threads or not self.scan_results.empty()) and self.winfo_exists():
            self.after(POLL_INTERVAL_MS, self._poll_scans)

    def clear_inputs(self):
        for row in self.rows.values():
            row.destroy()
        self.rows = {}

    def browse_output(self):
        path = filedialog.askdirectory(parent=self, title='Select output folder')
        if path:
            self.set_output_folder(path)

    def set_output_folder(self, path):
        self.output_entry.delete(0, 'end')
        self.output_entry.insert(0, os.path.normpath(path))

    def log(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    # conversion

    def start_conversion(self):
        if self.busy:
            return
        if not self.rows:
            self.log('Please add at least one input file or folder.')
            return
        if not self.output_folder:
            self.log('Please choose an output folder, or enter a relative path to use next to each input.')
            return

        # one conversion per chosen part ("All": each part; no parts: the whole input)
        conversions = [(path, label, source_kwargs, resolve_output_folder(self.output_folder, path))
                       for path, row in self.rows.items()
                       for label, source_kwargs in row.get_parts()]

        self.stop_requested.clear()
        self._set_running(True)
        self.progress.set(0)
        self.worker = threading.Thread(
            target=self._convert_all,
            args=(conversions, OUTPUT_FORMATS[self.format_menu.get()]),
            daemon=True)
        self.worker.start()
        self.after(POLL_INTERVAL_MS, self._poll_messages)

    def request_stop(self):
        self.stop_requested.set()
        self.stop_button.configure(state='disabled')
        self.log('Stop requested: finishing the current conversion, then stopping.')

    def _convert_all(self, conversions, output_format):
        # runs in the conversion thread: only communicates with the GUI through self.messages
        messages = self.messages
        handler = _QueueLogHandler(messages)
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        logging.getLogger('ome_zarr').setLevel(logging.WARNING)

        nconversions = len(conversions)
        nconverted = 0
        try:
            for index, (input_path, label, source_kwargs, output_folder) in enumerate(conversions):
                if self.stop_requested.is_set():
                    messages.put(('log', 'Stopped by user.'))
                    break
                part = f' ({label})' if label else ''
                messages.put(('log', f'[{index + 1}/{nconversions}] Converting {input_path}{part}'))
                try:
                    result = convert(input_path, output_folder, output_format=output_format, **source_kwargs)
                    for output in json.loads(result or '[]'):
                        messages.put(('log', f"    -> {output.get('full_path', output)}"))
                    nconverted += 1
                except Exception as error:
                    # the cause is logged by convert() above
                    messages.put(('log', f'    FAILED: {error}'))
                messages.put(('progress', (index + 1) / nconversions))
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            messages.put(('done', f'Finished: {nconverted}/{nconversions} converted successfully.'))

    def _poll_messages(self):
        done = False
        try:
            while True:
                kind, content = self.messages.get_nowait()
                if kind == 'log':
                    self.log(content)
                elif kind == 'progress':
                    self.progress.set(content)
                elif kind == 'done':
                    self.log(content)
                    done = True
        except queue.Empty:
            pass
        if done:
            self._set_running(False)
            if self.close_when_done:
                self.winfo_toplevel().destroy()
        elif self.winfo_exists():
            self.after(POLL_INTERVAL_MS, self._poll_messages)

    def _set_running(self, running):
        state = 'disabled' if running else 'normal'
        for widget in (self.convert_button, self.add_files_button, self.add_folder_button, self.clear_button,
                       self.output_entry, self.output_button, self.format_menu):
            widget.configure(state=state)
        for row in self.rows.values():
            row.set_state(state)
        self.stop_button.configure(state='normal' if running else 'disabled')

    def close(self):
        """Close the window holding this frame; while converting, close after the current conversion."""
        if self.busy:
            self.close_when_done = True
            self.request_stop()
            self.log('The window closes when the current conversion is finished.')
        else:
            self.winfo_toplevel().destroy()


class ConverterWindow(customtkinter.CTkToplevel):
    """The converter interface in its own window, as a child of an application window."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title(TITLE)
        self.geometry('960x640')
        self.transient(master)
        self.frame = ConverterFrame(self, fg_color='transparent')
        self.frame.pack(fill='both', expand=True)
        self.protocol('WM_DELETE_WINDOW', self.frame.close)
        self.lift()
        self.focus_force()


def main():
    """Standalone application: ``biomero-converter-gui``."""
    customtkinter.set_appearance_mode('system')
    app = customtkinter.CTk()
    app.title(TITLE)
    app.geometry('960x640')
    frame = ConverterFrame(app, fg_color='transparent')
    frame.pack(fill='both', expand=True)
    app.protocol('WM_DELETE_WINDOW', frame.close)
    app.mainloop()


if __name__ == '__main__':
    main()
