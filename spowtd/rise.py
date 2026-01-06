"""Determine rising curves"""

import logging

import numpy as np
import pandas as pd

from spowtd.fit_offsets import (
    assemble_weighted_linear_system,
    get_series_offsets,
    build_connected_head_mapping,
)


LOG = logging.getLogger('spowtd.rise')


def find_rise_offsets(connection, reference_zeta_mm=None, recharge_error_weight=0):
    """Determine rising curves

    If recharge_error_weight is provided, an estimated variance-covariance matrix is
    assembled and used when solving the rise curve assembly problem.

    """
    cursor = connection.cursor()
    compute_rise_offsets(
        cursor, reference_zeta_mm, recharge_error_weight=recharge_error_weight
    )
    cursor.close()
    connection.commit()


def compute_rise_offsets(cursor, reference_zeta_mm, recharge_error_weight=0):
    """Compute storage offsets to populate rising_interval_zeta

    If a reference zeta is not None, the crossing-depth of this water level is used as
    the origin of the axis.  Otherwise, the highest water level in the longest assembled
    rise is used.

    """
    (epoch, zeta_mm) = [
        np.array(v, dtype='float64')
        for v in zip(
            *cursor.execute(
                """
         SELECT epoch, zeta_mm FROM water_level
         ORDER BY epoch"""
            )
        )
    ]
    assert np.isfinite(zeta_mm).all()

    cursor.execute(
        """
    SELECT s.start_epoch,
           s.thru_epoch,
           zi.start_epoch,
           zi.thru_epoch
    FROM storm AS s
    JOIN zeta_interval_storm AS zis
      ON s.start_epoch = zis.storm_start_epoch
    JOIN zeta_interval AS zi
      ON zi.start_epoch = zis.interval_start_epoch
    ORDER BY s.start_epoch"""
    )

    series = []
    series_outlier = []
    rain_intervals = []
    zeta_intervals = []
    for (
        storm_start_epoch,
        storm_thru_epoch,
        zeta_start_epoch,
        zeta_thru_epoch,
    ) in cursor.fetchall():
        rain_start = np.argwhere(epoch == storm_start_epoch)[0, 0]
        # Epoch associated with rainfall intensities are *start* epoch for the interval,
        # so the time slice that *starts* at the storm thru_epoch is not included.
        rain_stop = np.argwhere(epoch == storm_thru_epoch)[0, 0]
        zeta_start = np.argwhere(epoch == zeta_start_epoch)[0, 0]
        zeta_thru = np.argwhere(epoch == zeta_thru_epoch)[0, 0]
        cursor.execute(
            """
        SELECT total_depth_mm
        FROM storm_total_rain_depth
        WHERE storm_start_epoch = :storm_start_epoch""",
            {'storm_start_epoch': storm_start_epoch},
        )
        total_depth = cursor.fetchone()[0]
        assert zeta_thru > zeta_start
        zeta_seq = zeta_mm[zeta_start : zeta_thru + 1]
        assert len(zeta_seq) >= 2, 'A jump is defined by at least two zeta values'
        assert (np.diff(zeta_seq) > 0).all(), '{} is not strictly increasing'.format(
            zeta_seq
        )
        initial_zeta = zeta_seq[0]
        final_zeta = zeta_seq[-1]
        assert len(zeta_seq) > 0, 'empty sequence'  # pylint:disable=len-as-condition
        rain_intervals.append((rain_start, rain_stop))
        zeta_intervals.append((zeta_start, zeta_thru + 1))

        # SA! added: to standardize events to same WL range in cm and adjust the
        #            corresponding P with the same factor
        # SA! added set the length (in mm) and turn on and off manually
        stretch = 0
        length = 100
        if stretch == 1:
            factor = (abs(final_zeta - initial_zeta)) / length
            total_depth = total_depth / factor
            initial_zeta_original = initial_zeta
            final_zeta_original = final_zeta
            initial_zeta = ((final_zeta_original + initial_zeta_original) / 2) - (
                length / 2
            )
            final_zeta = ((final_zeta_original + initial_zeta_original) / 2) + (
                length / 2
            )

        series.append(
            (np.array((0, total_depth)), np.array((initial_zeta, final_zeta)))
        )

        # SA! The general approach of the changes included below is the following:
        # SA! First an additional series is created including information on the
        #     specific yield
        # SA! Second a dictionary is created from the series to hold the original index
        #     as a key throughout the elimination process
        # SA! Third is the removal process: 1) events that are excluded during the
        #     offset calculation are removed from the series,
        # SA! 2) unrealistic events (Sy>1) are removed, 3) outlier detection and
        #     removal, and 4) events excluded during recalculation of offsets are removed
        # SA! Create an ndarray that includes Sy to filter out rise events based on Sy
        series_outlier.append(
            (
                np.array((0, total_depth)),
                np.array((initial_zeta, final_zeta)),
                total_depth / (final_zeta - initial_zeta),
            )
        )

    delta_z_mm = get_head_step(cursor)
    # Solve for offsets
    indices, offsets, zeta_mapping = get_series_storage_offsets(
        series, delta_z_mm, recharge_error_weight=recharge_error_weight
    )

    # create a series_dictionary from the list to maintain the indices as a key value
    # throughout the elimination process from the series
    indices_outlier = list(range(len(series)))
    series_dictionary = {}
    for key in indices_outlier:
        for value in series:
            series_dictionary[key] = value
            series.remove(value)
            break
    series_dictionary

    if len(indices) != len(series):
        drop_1 = list(sorted(set(range(len(series))) - set(indices)))
        for element in sorted(drop_1, reverse=True):
            del series_dictionary[element]

    # SA! remove unrealistic and outlier rise events
    outliers_removal = 1
    if outliers_removal == 1:
        # SA! This part removes the unrealistic rise events by filtering for series_id
        #     with Sy > 1 and removing them.
        # SA! Create an ndarray with the index of all series_ids to remove
        drop_2 = np.empty((0), dtype=int)
        for index, (t, H, Sy) in enumerate(series_outlier):
            if Sy > 1:
                drop_2 = np.append(drop_2, index)

        # SA! loop over ndarray in reverse order to maintain index position while
        #     removing
        # SA! remove from variables: series, indices and zeta_mapping (via crossing)
        for k in sorted(drop_2, reverse=True):
            del series_dictionary[k]
            indices.remove(k)
            for discrete_zeta, crossings in zeta_mapping.items():
                for series_id, mean_crossing_depth_mm in crossings:
                    if k == series_id:
                        for i in range(len(crossings)):
                            if len(crossings) <= i:
                                continue
                            elif k in crossings[i]:
                                del crossings[i]

        # SA! This part identifies outliers at each zeta_mm increment based on the Sy
        #     values and adds the series_id of the outlier to a counting array
        # SA! At each zeta_mm increment dataframe with series_id and Sy value are created
        outlier_count = np.empty((0, 2), dtype=float)
        zeta_step_sy = []
        occurence_count = []
        for discrete_zeta, crossings in zeta_mapping.items():
            for series_id, mean_crossing_depth_mm in crossings:
                occurence_count.append(series_id)
                for index, (t, H, Sy) in enumerate(series_outlier):
                    if index == series_id:
                        for i in range(len(crossings)):
                            if index in crossings[i]:
                                specific_yield = np.array([series_id, Sy])
                zeta_step_sy.append(specific_yield.copy())
            df = pd.DataFrame(
                list(map(np.ravel, zeta_step_sy)), columns=['series_id', 'Sy']
            )
            zeta_step_sy = []

            # SA! At each zeta_mm increment determine Sy outliers and add the series_id
            #     to the counting array
            # SA! If there are less than 5 events on the zeta_m apply boxplot outlier
            #     detection
            # SA! if there are 5 or more apply the 95th and 5th percentile outlier
            #     detection
            if len(df) >= 5:
                highest_allowed = df['Sy'].quantile(0.95)
                lowest_allowed = df['Sy'].quantile(0.05)
            else:
                IQR = df['Sy'].quantile(0.75) - df['Sy'].quantile(0.25)
                highest_allowed = df['Sy'].quantile(0.75) + 1.5 * (IQR)
                lowest_allowed = df['Sy'].quantile(0.25) - 1.5 * (IQR)
            df_outliers = df[(df['Sy'] > highest_allowed) | (df['Sy'] < lowest_allowed)]
            outlier_count = np.append(outlier_count, df_outliers.to_numpy(), axis=0)

        # SA! This part creates an ndarray with the outlier series_id based on the count
        drop_3 = np.empty((0), dtype=int)
        for j in range(len(series_dictionary)):
            how_many = np.count_nonzero(outlier_count[:, 0] == j)
            total = np.array(occurence_count)
            if how_many / (np.count_nonzero(total[:] == j) + 0.000000001) > 1 / 3:
                drop_3 = np.append(drop_3, j)

        # SA! loop over ndarray in reverse order to maintain index position while
        #     removing
        # SA! remove from variables: series, indices and zeta_mapping (via crossing)
        for z in sorted(drop_3, reverse=True):
            del series_dictionary[z]
            indices.remove(z)
            for discrete_zeta, crossings in zeta_mapping.items():
                for series_id, mean_crossing_depth_mm in crossings:
                    if z == series_id:
                        for i in range(len(crossings)):
                            if len(crossings) <= i:
                                continue
                            elif z in crossings[i]:
                                del crossings[i]

        series = list(series_dictionary.values())

        # SA! This solves for the offsets again after removing unrealistic and outlier
        #     events from series.
        # SA! Creates a copy of zeta_mapping and indices because the original with
        #     removed events are used.
        indices_copy, offsets, zeta_mapping_copy = get_series_storage_offsets(
            series, delta_z_mm, recharge_error_weight=recharge_error_weight
        )

        # SA! If additional series_ids were removed from offsets (necessary overlap of
        #     events)
        # SA! Check this series_id and drop all zeta_mm from zeta_mappings with this
        #     series_id and from the indices.
        for key in zeta_mapping.keys() - zeta_mapping_copy.keys():
            zeta_mapping.pop(key, None)
        unique_ids = []
        if len(indices) != len(offsets):
            for discrete_zeta, crossings in zeta_mapping.items():
                for series_id, mean_crossing_depth_mm in crossings:
                    unique_ids.append(series_id)
            missing = list(sorted((set(indices)) - (set(unique_ids))))
            for ele in missing:
                indices.remove(ele)

    reference_zeta_off_grid = reference_zeta_mm is not None and not np.allclose(
        reference_zeta_mm % delta_z_mm, 0
    )
    if reference_zeta_off_grid:
        raise ValueError(
            f'Reference zeta {reference_zeta_mm} mm not evenly divisible by '
            f'zeta step {delta_z_mm} mm'
        )
    if reference_zeta_mm is not None:
        reference_index = int(reference_zeta_mm / delta_z_mm)
    else:
        reference_index = max(zeta_mapping.keys())

    mean_zero_crossing_depth_mm = np.array(
        [
            offsets[indices.index(series_id)] + depth_mean_mm
            for series_id, depth_mean_mm in zeta_mapping[reference_index]
        ]
    ).mean()

    for i, series_id in enumerate(indices):
        interval = zeta_intervals[series_id]
        del series_id
        cursor.execute(
            """
        INSERT INTO rising_interval (
          start_epoch, rain_depth_offset_mm)
        SELECT :start_epoch,
               :rain_depth_offset_mm""",
            {
                'start_epoch': epoch[interval[0]],
                'rain_depth_offset_mm': (offsets[i] - mean_zero_crossing_depth_mm),
            },
        )
        del interval

    for discrete_zeta, crossings in zeta_mapping.items():
        for series_id, mean_crossing_depth_mm in crossings:
            interval = zeta_intervals[series_id]
            del series_id
            cursor.execute(
                """
            INSERT INTO rising_interval_zeta (
              start_epoch, zeta_number,
              mean_crossing_depth_mm)
            SELECT :start_epoch,
                   :discrete_zeta,
                   :mean_crossing_depth_mm""",
                {
                    'start_epoch': epoch[interval[0]],
                    'discrete_zeta': discrete_zeta,
                    'mean_crossing_depth_mm': mean_crossing_depth_mm,
                },
            )
            del mean_crossing_depth_mm, interval
        del discrete_zeta, crossings
    cursor.close()


def assemble_rise_series(cursor):
    """Assemble rise series

    Returns:
      series:  A list of pairs of arrays
               ((0, total_depth), (initial_zeta, final_zeta)),
               each representing a distinct rise event.
      epoch:  Time vector as seconds since the UNIX epoch
      zeta_intervals:  A list of (start, thru) indices in the epoch vector, one for
                       each series

    """
    epoch, zeta_mm = [
        np.array(v, dtype='float64')
        for v in zip(
            *cursor.execute(
                """
            SELECT epoch, zeta_mm FROM water_level
            ORDER BY epoch"""
            )
        )
    ]
    assert np.isfinite(zeta_mm).all()
    cursor.execute(
        """
    SELECT s.start_epoch,
           s.thru_epoch,
           zi.start_epoch,
           zi.thru_epoch
    FROM storm AS s
    JOIN zeta_interval_storm AS zis
      ON s.start_epoch = zis.storm_start_epoch
    JOIN zeta_interval AS zi
      ON zi.start_epoch = zis.interval_start_epoch
    ORDER BY s.start_epoch"""
    )
    series = []
    rain_intervals = []
    zeta_intervals = []
    for (
        storm_start_epoch,
        storm_thru_epoch,
        zeta_start_epoch,
        zeta_thru_epoch,
    ) in cursor.fetchall():
        rain_start = np.argwhere(epoch == storm_start_epoch)[0, 0]
        # Epoch associated with rainfall intensities are *start* epoch for the interval,
        # so the time slice that *starts* at the storm thru_epoch is not included.
        rain_stop = np.argwhere(epoch == storm_thru_epoch)[0, 0]
        zeta_start = np.argwhere(epoch == zeta_start_epoch)[0, 0]
        zeta_thru = np.argwhere(epoch == zeta_thru_epoch)[0, 0]
        cursor.execute(
            """
        SELECT total_depth_mm
        FROM storm_total_rain_depth
        WHERE storm_start_epoch = :storm_start_epoch""",
            {'storm_start_epoch': storm_start_epoch},
        )
        total_depth = cursor.fetchone()[0]
        assert zeta_thru > zeta_start
        zeta_seq = zeta_mm[zeta_start : zeta_thru + 1]
        assert len(zeta_seq) >= 2, 'A jump is defined by at least two zeta values'
        assert (np.diff(zeta_seq) > 0).all(), f'{zeta_seq} is not strictly increasing'
        initial_zeta = zeta_seq[0]
        final_zeta = zeta_seq[-1]
        assert len(zeta_seq) > 0, 'empty sequence'  # pylint:disable=len-as-condition
        rain_intervals.append((rain_start, rain_stop))
        zeta_intervals.append((zeta_start, zeta_thru + 1))
        series.append(
            (np.array((0, total_depth)), np.array((initial_zeta, final_zeta)))
        )

    del zeta_mm
    return series, epoch, zeta_intervals


def get_head_step(cursor):
    """Get grid interval in mm from database cursor

    If grid_interval_mm is not yet populated in zeta_grid, ValueError is raised

    """
    try:
        delta_z_mm = cursor.execute(
            """
        SELECT (grid_interval_mm) FROM zeta_grid
        """
        ).fetchone()[0]
    except TypeError:
        raise ValueError(  # pylint: disable=raise-missing-from
            'Discrete water level interval not yet set'
        )
    return delta_z_mm


def get_series_storage_offsets(series_list, head_step, recharge_error_weight=0):
    """Find a storage offset that minimizes difference in head crossing times

    This function is used in assembly of rise curves.  See further documentation under
    get_series_offsets.

    If recharge_error_weight is given, an estimated variance-covariance matrix is
    assembled and used when solving the rise curve assembly problem.

    """
    head_mapping, index_mapping = build_connected_head_mapping(series_list, head_step)
    return get_series_offsets(
        head_mapping,
        index_mapping,
        recharge_error_weight=recharge_error_weight,
    )


def get_rise_covariance(connection, recharge_error_weight):
    """Use database connection to build covariance of rise event errors"""
    cursor = connection.cursor()
    series, _, _ = assemble_rise_series(cursor)
    head_step = get_head_step(cursor)
    cursor.close()
    head_mapping, index_mapping = build_connected_head_mapping(series, head_step)
    return assemble_rise_covariance(
        head_mapping, index_mapping, series, head_step, recharge_error_weight
    )


def assemble_rise_covariance(
    head_mapping, index_mapping, series, head_step, recharge_error_weight=1e3
):
    """Assemble covariance of rise event errors

    Creates and populates matrix omega representing the errors in the overdetermined
    system Ax = b for the rise curve assembly problem.

    head_mapping: maps head_id to a sequence of (series_id, time) pairs indicating when
                  each series crossed that head.
    index_mapping: maps series_id to the index of the series in series_list.
    series: list of (depth, rise) pairs where depth is a (0, storm_depth) duple and rise
            is a (zeta_initial, zeta_final) duple.

    head_step: factor by which to multiply integer head_id to get water level zeta.
    recharge_error_weight: ratio of error induced by recharge depth mismeasurement to
                           intrinsic error along the storage axis.  If zero or None, the
                           K x K identity matrix is returned where K is the total number
                           of equations.

    The covariance of rise event errors is a symmetric, positive semidefinite matrix
    with shape (n_equations, n_equations) characterizing the covariance among errors in
    each equation of the rise curve assembly problem.

    Recharge_error_weight is the relative weight for errors arising from mismeasurement
    of recharge depth.  The identity matrix, divided by recharge_error_weight, is added
    to the rise event error covariance to make it positive definite.

    """
    check_rise_head_mapping(head_mapping, series, index_mapping, head_step)
    _, _, Omega = assemble_weighted_linear_system(
        head_mapping,
        index_mapping,
        recharge_error_weight=recharge_error_weight,
    )
    return Omega


def check_rise_head_mapping(head_mapping, series, index_mapping, head_step):
    """Check head mapping for rise curve assembly

    Verify that each rise contains exactly two depths that span the associated water
    level.

    """
    for head_id, series_at_head in head_mapping.items():
        zeta = head_id * head_step
        for series_id, depth_at_head in series_at_head:
            depth, rise = series[index_mapping[series_id]]
            assert len(depth) == 2
            assert len(rise) == 2
            assert depth[0] == 0, f'depth[0] = {depth[0]}'
            assert depth_at_head >= 0, depth_at_head
            assert depth_at_head <= depth[1], f'{depth_at_head} > {depth[1]}'
            assert rise[0] <= zeta <= rise[1], f'Rise {rise} does not span {zeta}'
            del depth, rise, series_id, depth_at_head
        del head_id, zeta, series_at_head
