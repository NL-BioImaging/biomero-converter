import json
import logging
import os.path
import queue
import threading
from tkinter import filedialog

import customtkinter

from biomero_converter.converter import convert
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

TITLE = f'biomero-converter {VERSION}'
POLL_INTERVAL_MS = 100


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


class ConverterFrame(customtkinter.CTkFrame):
    """The converter interface: inputs, output folder, options, log and progress.

    Conversions run in a background thread, one input after the other; a failed input is
    logged and the others continue.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.inputs = []                    # input paths (files or folders)
        self.output_folder = None
        self.messages = queue.Queue()       # conversion thread -> GUI thread
        self.worker = None
        self.stop_requested = threading.Event()
        self.close_when_done = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

        customtkinter.CTkLabel(
            self,
            text='Convert microscopy files to OME-Zarr or OME-TIFF.\n'
                 'Supports Leica, Incucyte, iSyntax, MIRAX, DICOM, image .db, (OME-)TIFF and OME-Zarr.',
            anchor='center').grid(row=0, column=0, sticky='new', pady=5, padx=10)

        # inputs
        input_frame = customtkinter.CTkFrame(self)
        input_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=10)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_rowconfigure(1, weight=1)
        customtkinter.CTkLabel(input_frame, text='Input files / folders', anchor='w').grid(
            row=0, column=0, sticky='w', pady=(5, 0), padx=5)
        self.input_box = customtkinter.CTkTextbox(input_frame, height=120, wrap='none')
        self.input_box.grid(row=1, column=0, rowspan=4, sticky='nsew', pady=5, padx=5)
        self.input_box.configure(state='disabled')
        self.add_files_button = customtkinter.CTkButton(input_frame, text='Add files', command=self.browse_files)
        self.add_files_button.grid(row=1, column=1, sticky='ew', pady=5, padx=5)
        self.add_folder_button = customtkinter.CTkButton(input_frame, text='Add folder', command=self.browse_folder)
        self.add_folder_button.grid(row=2, column=1, sticky='ew', pady=5, padx=5)
        self.clear_button = customtkinter.CTkButton(input_frame, text='Clear', command=self.clear_inputs)
        self.clear_button.grid(row=3, column=1, sticky='ew', pady=5, padx=5)

        # output folder
        output_frame = customtkinter.CTkFrame(self)
        output_frame.grid(row=2, column=0, sticky='ew', pady=5, padx=10)
        output_frame.grid_columnconfigure(0, weight=1)
        self.output_label = customtkinter.CTkLabel(output_frame, text='Output folder: (none selected)', anchor='w')
        self.output_label.grid(row=0, column=0, sticky='w', pady=5, padx=5)
        self.output_button = customtkinter.CTkButton(output_frame, text='Browse', command=self.browse_output)
        self.output_button.grid(row=0, column=1, sticky='ew', pady=5, padx=5)

        # options
        options = customtkinter.CTkFrame(self)
        options.grid(row=3, column=0, sticky='ew', pady=5, padx=10)
        options.grid_columnconfigure((1, 3, 5), weight=1)
        customtkinter.CTkLabel(options, text='Output format:', anchor='w').grid(
            row=0, column=0, sticky='w', pady=5, padx=5)
        self.format_menu = customtkinter.CTkOptionMenu(options, values=list(OUTPUT_FORMATS))
        self.format_menu.set(DEFAULT_FORMAT_LABEL)
        self.format_menu.grid(row=0, column=1, sticky='ew', pady=5, padx=5)
        customtkinter.CTkLabel(options, text='Incucyte plate ID:', anchor='w').grid(
            row=0, column=2, sticky='w', pady=5, padx=5)
        self.plate_entry = customtkinter.CTkEntry(options, placeholder_text='all plates')
        self.plate_entry.grid(row=0, column=3, sticky='ew', pady=5, padx=5)
        customtkinter.CTkLabel(options, text='Leica image UUID:', anchor='w').grid(
            row=0, column=4, sticky='w', pady=5, padx=5)
        self.uuid_entry = customtkinter.CTkEntry(options, placeholder_text='all images')
        self.uuid_entry.grid(row=0, column=5, sticky='ew', pady=5, padx=5)

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
        for path in paths:
            path = os.path.normpath(path)
            if path not in self.inputs:
                self.inputs.append(path)
        self._show_inputs()

    def clear_inputs(self):
        self.inputs = []
        self._show_inputs()

    def _show_inputs(self):
        self.input_box.configure(state='normal')
        self.input_box.delete('0.0', 'end')
        self.input_box.insert('0.0', '\n'.join(self.inputs))
        self.input_box.configure(state='disabled')

    def browse_output(self):
        path = filedialog.askdirectory(parent=self, title='Select output folder')
        if path:
            self.set_output_folder(path)

    def set_output_folder(self, path):
        self.output_folder = os.path.normpath(path)
        self.output_label.configure(text=f'Output folder: {self.output_folder}')

    def log(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    # conversion

    def start_conversion(self):
        if self.busy:
            return
        if not self.inputs:
            self.log('Please add at least one input file or folder.')
            return
        if not self.output_folder:
            self.log('Please select an output folder.')
            return

        source_kwargs = {}
        plate_id = self.plate_entry.get().strip()
        image_uuid = self.uuid_entry.get().strip()
        if plate_id:
            source_kwargs['plate_id'] = plate_id
        if image_uuid:
            source_kwargs['image_uuid'] = image_uuid

        self.stop_requested.clear()
        self._set_running(True)
        self.progress.set(0)
        self.worker = threading.Thread(
            target=self._convert_all,
            args=(list(self.inputs), self.output_folder, OUTPUT_FORMATS[self.format_menu.get()], source_kwargs),
            daemon=True)
        self.worker.start()
        self.after(POLL_INTERVAL_MS, self._poll_messages)

    def request_stop(self):
        self.stop_requested.set()
        self.stop_button.configure(state='disabled')
        self.log('Stop requested: finishing the current input, then stopping.')

    def _convert_all(self, input_paths, output_folder, output_format, source_kwargs):
        # runs in the conversion thread: only communicates with the GUI through self.messages
        messages = self.messages
        handler = _QueueLogHandler(messages)
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        logging.getLogger('ome_zarr').setLevel(logging.WARNING)

        ninputs = len(input_paths)
        nconverted = 0
        try:
            for index, input_path in enumerate(input_paths):
                if self.stop_requested.is_set():
                    messages.put(('log', 'Stopped by user.'))
                    break
                messages.put(('log', f'[{index + 1}/{ninputs}] Converting {input_path}'))
                try:
                    result = convert(input_path, output_folder, output_format=output_format, **source_kwargs)
                    for output in json.loads(result or '[]'):
                        messages.put(('log', f"    -> {output.get('full_path', output)}"))
                    nconverted += 1
                except Exception as error:
                    # the cause is logged by convert() above
                    messages.put(('log', f'    FAILED: {error}'))
                messages.put(('progress', (index + 1) / ninputs))
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            messages.put(('done', f'Finished: {nconverted}/{ninputs} input(s) converted successfully.'))

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
                       self.output_button, self.format_menu, self.plate_entry, self.uuid_entry):
            widget.configure(state=state)
        self.stop_button.configure(state='normal' if running else 'disabled')

    def close(self):
        """Close the window holding this frame; while converting, close after the current input."""
        if self.busy:
            self.close_when_done = True
            self.request_stop()
            self.log('The window closes when the current input is finished.')
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
