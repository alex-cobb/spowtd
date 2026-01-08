# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for data loading module"""


def test_load_sample_data(loaded_connection):
    """Spowtd correctly loads sample data"""
    cursor = loaded_connection.cursor()
    # XXX Check loaded data
    cursor.close()
