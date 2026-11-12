"""Desktop GUI for biomero-converter (optional: ``pip install biomero-converter[gui]``).

- ``ConverterFrame``: the converter interface as a widget, to place in any customtkinter/tkinter window.
- ``ConverterWindow``: the interface in its own child window, for applications that host it.
- ``main()``: the standalone application (``biomero-converter-gui``).
"""

try:
    import customtkinter  # noqa: F401
except ImportError as error:
    raise ImportError('The biomero-converter GUI needs customtkinter: '
                      'pip install "biomero-converter[gui]"') from error

from biomero_converter.gui.app import ConverterFrame, ConverterWindow, main

__all__ = ['ConverterFrame', 'ConverterWindow', 'main']
