# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Setting site curvature in database"""


def set_curvature(connection, curvature_m_km2):
    """Set site curvature"""
    cursor = connection.cursor()
    cursor.execute(
        """
    INSERT INTO curvature (curvature_m_km2)
    VALUES (?)""",
        (curvature_m_km2,),
    )
    cursor.close()
