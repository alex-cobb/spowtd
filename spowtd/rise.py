"""Determine rising curves

"""

import logging

import numpy as np

from spowtd.fit_offsets import get_series_time_offsets


LOG = logging.getLogger('spowtd.rise')


def find_rise_offsets(
        connection,
        reference_zeta_mm=None):
    """Determine rising curves

    """
    cursor = connection.cursor()
    compute_rise_offsets(cursor, reference_zeta_mm)
    cursor.close()
    connection.commit()


def compute_rise_offsets(cursor,
                         reference_zeta_mm):
    """Compute time offsets to populate rising_interval_zeta

    If a reference zeta is provided, the crossing-depth of this water
    level is used as the origin of the axis.  Otherwise, the highest
    water level in the longest assembled rise is used.

    """
    (epoch,
     zeta_mm) = [
         np.array(v, dtype='float64') for v in
         zip(*cursor.execute("""
         SELECT epoch, zeta_mm FROM water_level
         ORDER BY epoch"""))]
    assert np.isfinite(zeta_mm).all()

    cursor.execute("""
    SELECT s.start_epoch,
           s.thru_epoch,
           zi.start_epoch,
           zi.thru_epoch
    FROM storm AS s
    JOIN zeta_interval_storm AS zis
      ON s.start_epoch = zis.storm_start_epoch
    JOIN zeta_interval AS zi
      ON zi.start_epoch = zis.interval_start_epoch
    ORDER BY s.start_epoch""")

    series = []
    rain_intervals = []
    zeta_intervals = []
    for (storm_start_epoch, storm_thru_epoch,
         zeta_start_epoch, zeta_thru_epoch) in cursor.fetchall():
        rain_start = np.argwhere(epoch == storm_start_epoch)[0, 0]
        # Epoch associated with rainfall intensities are *start*
        # epoch for the interval, so the time slice that *starts*
        # at the storm thru_epoch is not included.
        rain_stop = np.argwhere(epoch == storm_thru_epoch)[0, 0]
        zeta_start = np.argwhere(epoch == zeta_start_epoch)[0, 0]
        zeta_thru = np.argwhere(epoch == zeta_thru_epoch)[0, 0]
        cursor.execute("""
        SELECT total_depth_mm
        FROM storm_total_rain_depth
        WHERE storm_start_epoch = :storm_start_epoch""",
                       {'storm_start_epoch': storm_start_epoch})
        total_depth = cursor.fetchone()[0]
        assert zeta_thru > zeta_start
        zeta_seq = zeta_mm[zeta_start:zeta_thru + 1]
        assert len(zeta_seq) >= 2, (
            'A jump is defined by at least two zeta values')
        assert (np.diff(zeta_seq) > 0).all(), (
            '{} is not strictly increasing'.format(zeta_seq))
        initial_zeta = zeta_seq[0]
        final_zeta = zeta_seq[-1]
        assert len(zeta_seq) > 0, (  # pylint:disable=len-as-condition
            'empty sequence')
        rain_intervals.append((rain_start, rain_stop))
        zeta_intervals.append((zeta_start, zeta_thru + 1))
        series.append((np.array((0, total_depth)),
                       np.array((initial_zeta, final_zeta))))

    try:
        delta_z_mm = cursor.execute("""
        SELECT (grid_interval_mm) FROM zeta_grid
        """).fetchone()[0]
    except TypeError:
        raise ValueError(
            "Discrete water level interval not yet set")

    # Solve for offsets
    indices, offsets, zeta_mapping = get_series_time_offsets(
        series, delta_z_mm)

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
        reference_index = max(zeta_mapping.keys())

    mean_zero_crossing_depth_mm = np.array(
        [offsets[indices.index(series_id)] + depth_mean_mm
         for series_id, depth_mean_mm
         in zeta_mapping[reference_index]]).mean()

    for i, series_id in enumerate(indices):
        interval = zeta_intervals[series_id]
        del series_id
        cursor.execute("""
        INSERT INTO rising_interval (
          start_epoch, rain_depth_offset_mm)
        SELECT :start_epoch,
               :rain_depth_offset_mm""",
                       {'start_epoch': epoch[interval[0]],
                        'rain_depth_offset_mm':
                        (offsets[i] -
                         mean_zero_crossing_depth_mm)})
        del interval

    for discrete_zeta, crossings in zeta_mapping.items():
        for series_id, mean_crossing_depth_mm in crossings:
            interval = zeta_intervals[series_id]
            del series_id
            cursor.execute("""
            INSERT INTO rising_interval_zeta (
              start_epoch, zeta_number,
              mean_crossing_depth_mm)
            SELECT :start_epoch,
                   :discrete_zeta,
                   :mean_crossing_depth_mm""",
                           {'start_epoch': epoch[interval[0]],
                            'discrete_zeta': discrete_zeta,
                            'mean_crossing_depth_mm':
                            mean_crossing_depth_mm})
            del mean_crossing_depth_mm, interval
        del discrete_zeta, crossings
    cursor.close()
