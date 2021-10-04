"""Determine recession curves

"""

import logging

import numpy as np

from spowtd.fit_offsets import get_series_time_offsets


LOG = logging.getLogger('spowtd.recession')


def find_recession_offsets(
        connection,
        reference_zeta_mm=None):
    """Determine recession curves

    """
    cursor = connection.cursor()
    compute_offsets(cursor, reference_zeta_mm)
    cursor.close()


def compute_offsets(cursor, reference_zeta_mm):
    """Compute time offsets to populate recession_zeta

    If a reference zeta is provided, the crossing-time of this water
    level is used as the orign of the axis.  Otherwise, the highest
    water level in the longest assembled recession is used.

    """
    (epoch,
     zeta_mm) = [
         np.array(v, dtype='float64') for v in
         zip(*cursor.execute("""
         SELECT epoch, zeta_mm FROM water_level
         ORDER BY epoch"""))]
    assert np.isfinite(zeta_mm).all()
    cursor.execute("""
    SELECT start_epoch,
           thru_epoch
    FROM zeta_interval
    WHERE interval_type = 'interstorm'
    ORDER BY start_epoch""")
    series = []
    for i, (interval_start_epoch,
            interval_thru_epoch) in enumerate(cursor):
        indices = np.argwhere((epoch >= interval_start_epoch) &
                              (epoch <= interval_thru_epoch))[:, 0]
        assert epoch[indices][0] == interval_start_epoch, (
            '{} != {}'.format(epoch[indices][0], interval_start_epoch))
        assert epoch[indices][-1] == interval_thru_epoch, (
            '{} != {}'.format(epoch[indices][-1], interval_thru_epoch))
        series.append((epoch[indices], zeta_mm[indices]))
        del indices
        del i, interval_start_epoch, interval_thru_epoch

    try:
        delta_z_mm = cursor.execute("""
        SELECT (grid_interval_mm) FROM zeta_grid
        """).fetchone()[0]
    except TypeError:
        raise ValueError(
            "Discrete water level interval not yet set")

    (indices,
     offsets,
     head_mapping) = get_series_time_offsets(
         series,
         delta_z_mm)

    reference_zeta_off_grid = (
        reference_zeta_mm is not None and
        not np.allclose(reference_zeta_mm % delta_z_mm, 0))
    if reference_zeta_off_grid:
        raise ValueError(
            'Reference zeta {} mm not evenly divisible by '
            'zeta step {} mm'.format(reference_zeta_mm, delta_z_mm))
    if reference_zeta_mm is not None:
        reference_index = int(reference_zeta_mm / delta_z_mm)
    else:
        reference_index = max(head_mapping.keys())

    mean_zero_crossing_time_s = np.array(
        [offsets[indices.index(series_id)] + time_mean_s
         for series_id, time_mean_s
         in head_mapping[reference_index]]).mean()

    for i in range(len(indices)):
        series_id = indices[i]
        interval_epoch = series[series_id][0]
        del series_id
        cursor.execute("""
        SELECT EXISTS (
          SELECT 1 FROM zeta_interval
          WHERE start_epoch = ?
          AND interval_type = 'interstorm'
        )""", (interval_epoch[0],))
        fk_exists = cursor.fetchone()[0]
        assert fk_exists, interval_epoch[0]
        cursor.execute("""
        INSERT INTO recession_interval (
          start_epoch, time_offset)
        SELECT :start_epoch,
               :time_offset_s""",
                       {'start_epoch': interval_epoch[0],
                        'time_offset_s': (
                            offsets[i] -
                            mean_zero_crossing_time_s)})
        del interval_epoch

    for discrete_zeta, crossings in head_mapping.items():
        for series_id, mean_crossing_time_s in crossings:
            interval_epoch = series[series_id][0]
            del series_id
            cursor.execute("""
            INSERT INTO recession_interval_zeta (
              start_epoch, zeta_number,
              mean_crossing_time)
            SELECT :start_epoch,
                   :discrete_zeta,
                   :mean_crossing_time_s""",
                           {'start_epoch': interval_epoch[0],
                            'discrete_zeta': discrete_zeta,
                            'mean_crossing_time_s': mean_crossing_time_s})
            del mean_crossing_time_s, interval_epoch
        del discrete_zeta, crossings
