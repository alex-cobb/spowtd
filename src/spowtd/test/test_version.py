# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test for valid version string"""

import spowtd
import pathlib


def test_version_matches_file():
    """Test that package version matches version string in VERSION.txt"""
    root_dir = pathlib.Path(__file__).parents[2]
    version_file = root_dir / 'spowtd' / 'VERSION.txt'

    if version_file.exists():
        expected = version_file.read_text().strip()
        assert spowtd.__version__ == expected, 'Version matches file'
    else:
        # Running from installed package
        assert spowtd.__version__ != 'unknown'
