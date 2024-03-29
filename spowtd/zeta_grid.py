"""Populate water level grid

"""

import math


def populate_zeta_grid(connection, grid_interval_mm):
    """Populate uniform grid on zeta"""
    cursor = connection.cursor()
    cursor.execute(
        """
    INSERT INTO zeta_grid (grid_interval_mm) VALUES (?)
    """,
        (grid_interval_mm,),
    )
    cursor.execute(
        """
    SELECT min(zeta_mm), max(zeta_mm)
    FROM water_level"""
    )
    zeta_bounds = cursor.fetchone()
    cursor.executemany(
        """
    INSERT INTO discrete_zeta (zeta_number)
    VALUES (?)""",
        [
            (zn,)
            for zn in range(
                int(math.floor(zeta_bounds[0] / grid_interval_mm)),
                int(math.ceil(zeta_bounds[1] / grid_interval_mm)),
            )
        ],
    )
    cursor.close()
