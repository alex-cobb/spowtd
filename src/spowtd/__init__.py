"""Spowtd - scalar parameterization of water table dynamics"""

import importlib.resources
import pathlib


def _get_version():
    try:
        return (
            importlib.resources.files(__package__)
            .joinpath('VERSION.txt')
            .read_text()
            .strip()
        )
    except Exception:
        # Fallback
        version_file = pathlib.Path(__file__).parent / 'VERSION.txt'
        if version_file.exists():
            return version_file.read_text().strip()
        return 'unknown'


__version__ = _get_version()
