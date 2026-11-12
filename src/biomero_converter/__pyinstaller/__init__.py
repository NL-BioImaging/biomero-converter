"""PyInstaller support: registered as a ``pyinstaller40`` entry point in pyproject.toml,
so applications that bundle biomero-converter with PyInstaller pick up its hook automatically."""

import os


def get_hook_dirs():
    return [os.path.dirname(__file__)]
