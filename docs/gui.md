# Desktop GUI

biomero-converter includes an optional desktop application for converting files without the command line.
It runs on Windows, macOS and Linux.

## Install and start

```sh
pip install "biomero-converter[gui]"
biomero-converter-gui
```

The `[gui]` extra installs [customtkinter](https://github.com/TomSchimansky/CustomTkinter). The command line,
Docker image and Python API do not need it.

Tk must be available in the Python installation:

- Windows: included with the python.org and conda installers.
- macOS: use Python from python.org or conda (Tk 8.6).
- Linux: install your distribution's Tk package, e.g. `sudo apt install python3-tk`, or use conda.

## Use

1. **Add files** or **Add folder** (folder inputs such as OME-Zarr or DICOM folders); several inputs can be queued.
2. **Browse** to choose the output folder.
3. Choose the **output format** (default: OME-Zarr 0.5, see `DEFAULT_OUTPUT_FORMAT` in [Convert](convert.md)).
   Optionally enter an **Incucyte plate ID** (default: all plates) or a **Leica image UUID** (default: all images).
4. **Convert.** The inputs are converted one after the other; the log shows each output path. If an input fails,
   the error is logged and the next input is converted. **Stop after current** stops after the input being converted.

Closing the window during a conversion stops after the current input and then closes.

## Use in another application

The interface is also available as widgets, so other customtkinter/tkinter applications can host it:

```python
from biomero_converter.gui import ConverterFrame, ConverterWindow

ConverterWindow(master)             # the interface in its own child window
frame = ConverterFrame(parent)      # or as a widget inside an existing window
frame.add_inputs(['path/to/file.lif'])
frame.set_output_folder('path/to/output')
```

For applications frozen with PyInstaller, biomero-converter provides a PyInstaller hook (registered automatically)
that bundles the data files and libraries it needs at runtime.
