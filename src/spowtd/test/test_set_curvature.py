# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for setting site curvature"""

import spowtd.set_curvature as set_curvature_mod


def test_set_curvature(classified_connection):
    """Set site curvature"""
    ref_curvature = 9.876
    set_curvature_mod.set_curvature(classified_connection, ref_curvature)
    cursor = classified_connection.cursor()
    cursor.execute('SELECT curvature_m_km2 FROM curvature')
    curvature_m_km2 = cursor.fetchone()[0]
    assert curvature_m_km2 == ref_curvature
    cursor.close()
