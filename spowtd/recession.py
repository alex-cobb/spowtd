"""Determine recession curves

"""

import math
import logging

import numpy as np

from spowtd.fit_offsets import get_series_time_offsets


LOG = logging.getLogger('spowtd.recession')


def find_recession_offsets(
        connection,
        delta_z_mm=1.0):
    """Determine recession curves

    """
    cursor = connection.cursor()
    populate_zeta_grid(cursor, delta_z_mm)
    compute_offsets(cursor)
    cursor.close()


def populate_zeta_grid(cursor, grid_interval_mm):
    """Populate uniform grid on zeta

    """
    cursor.execute("""
    INSERT INTO zeta_grid (grid_interval_mm) VALUES (?)
    """, (grid_interval_mm,))
    cursor.execute("""
    SELECT min(zeta_mm), max(zeta_mm)
    FROM water_level""")
    zeta_bounds = cursor.fetchone()
    cursor.executemany("""
    INSERT INTO discrete_zeta (zeta_number)
    VALUES (?)""", [(zn,) for zn in range(
        int(math.floor(zeta_bounds[0] / grid_interval_mm)),
        int(math.ceil(zeta_bounds[1] / grid_interval_mm)))])


def compute_offsets(cursor):
    """Compute time offsets to populate recession_zeta for location

    """
    (epoch,
     zeta_mm) = [
         np.array(v, dtype='float64') for v in
         zip(*cursor.execute("""
         SELECT epoch, zeta_mm FROM water_level
         ORDER BY zeta_mm"""))]
    assert np.isfinite(zeta_mm).all()
    cursor.execute("""
    SELECT start_epoch,
           thru_epoch
    FROM zeta_interval
    WHERE interval_type = 'interstorm'
    ORDER BY start_epoch""")
    series_indices = []
    for i, (interval_start_epoch,
            interval_thru_epoch) in enumerate(cursor):
        indices = np.argwhere((epoch >= interval_start_epoch) &
                              (epoch <= interval_thru_epoch))[:, 0]
        series_indices.append(indices)
        del indices
        del i, interval_start_epoch, interval_thru_epoch

    indices = None
    elapsed_time_s = epoch - epoch[0]
    series = [(elapsed_time_s[indices], zeta_mm[indices])
              for indices in series_indices]
    del indices
    delta_z_mm = cursor.execute("""
    SELECT (grid_interval_mm) FROM zeta_grid
    """).fetchone()[0]
    (indices,
     offsets,
     head_mapping) = get_series_time_offsets(
         series,
         delta_z_mm)
    assert 0 in head_mapping, (
        'Centering requires that series crosses zeta = 0.0')

    mean_zero_crossing_time_s = np.array(
        [offsets[indices.index(series_id)] + time_mean_min
         for series_id, time_mean_min in head_mapping[0]]).mean()

    for i in range(len(indices)):
        series_id = indices[i]
        interval = series_indices[series_id]
        del series_id
        cursor.execute("""
        INSERT INTO recession_interval (
          start_epoch, time_offset)
        SELECT %(start_epoch)s,
               %(time_offset)s::interval""",
                       {'start_epoch': epoch[interval[0]],
                        'time_offset':
                        '{} s'.format(offsets[i] -
                                      mean_zero_crossing_time_s)})
        del interval

    for discrete_zeta, crossings in head_mapping.items():
        for series_id, mean_crossing_time_s in crossings:
            interval = series_indices[series_id]
            del series_id
            cursor.execute("""
            INSERT INTO recession_interval_zeta (
              start_epoch, zeta_number,
              mean_crossing_time)
            SELECT :start_epoch,
                   :discrete_zeta,
                   :mean_crossing_time""",
                           {'start_epoch': epoch[interval[0]],
                            'discrete_zeta': discrete_zeta,
                            'mean_crossing_time':
                            '{} s'.format(mean_crossing_time_s)})
            del mean_crossing_time_s, interval
        del discrete_zeta, crossings
