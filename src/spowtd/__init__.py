# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

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
