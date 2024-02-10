"""Determine rising curves

"""

import logging

import numpy as np

from spowtd.fit_offsets import (
    get_series_offsets,
    build_connected_head_mapping,
)


LOG = logging.getLogger('spowtd.rise')


def find_rise_offsets(
    connection, reference_zeta_mm=None, recharge_error_weight=0
):
    """Determine rising curves

    If recharge_error_weight is provided, an estimated variance-covariance
    matrix is assembled and used when solving the rise curve assembly problem.
    """
    cursor = connection.cursor()
    compute_rise_offsets(
        cursor, reference_zeta_mm, recharge_error_weight=recharge_error_weight
    )
    cursor.close()
    connection.commit()


def get_series_storage_offsets(
    series_list, head_step, recharge_error_weight=0
):
    """Find a storage offset that minimizes difference in head crossing times

    This function is used in assembly of rise curves.  See further
    documentation under get_series_offsets.

    If recharge_error_weight is given, an estimated variance-covariance matrix
    is assembled and used when solving the rise curve assembly problem.

    """
    head_mapping, index_mapping = build_connected_head_mapping(
        series_list, head_step
    )
    if not recharge_error_weight:
        return get_series_offsets(head_mapping, index_mapping)
    covariance = assemble_rise_covariance(
        head_mapping,
        index_mapping,
        series_list,
        head_step,
        recharge_error_weight,
    )
    return get_series_offsets(
        head_mapping, index_mapping, covariance=covariance
    )


def get_rise_covariance(connection, recharge_error_weight):
    """Use database connection to build covariance of rise event errors"""
    cursor = connection.cursor()
    series, _, _ = assemble_rise_series(cursor)
    head_step = get_head_step(cursor)
    cursor.close()
    head_mapping, index_mapping = build_connected_head_mapping(
        series, head_step
    )
    return assemble_rise_covariance(
        head_mapping, index_mapping, series, head_step, recharge_error_weight
    )


def assemble_rise_covariance(
    head_mapping, index_mapping, series, head_step, recharge_error_weight=1e3
):
    """Assemble covariance of rise event errors

    Creates and populates matrix omega representing the errors in the
    overdetermined system Ax = b for the rise curve assembly problem.

    head_mapping: maps head_id to a sequence of (series_id, time) pairs
                  indicating when each series crossed that head.
    index_mapping: maps series_id to the index of the series in series_list.
    series: list of (depth, rise) pairs where depth is a (0, storm_depth)
            duple and rise is a (zeta_initial, zeta_final) duple.
    head_step: factor by which to multiply integer head_id to get water
               level zeta.
    recharge_error_weight: ratio of error induced by recharge depth
                           mismeasurement to intrinsic error along the storage
                           axis.  If zero or None, the K x K identity matrix is
                           returned where K is the total number of equations.

    The covariance of rise event errors is a symmetric, positive semidefinite
    matrix with shape (n_equations, n_equations) characterizing the covariance
    among errors in each equation of the rise curve assembly problem.

    Recharge_error_weight is the relative weight for errors arising from
    mismeasurement of recharge depth.  The identity matrix, divided by
    recharge_error_weight, is added to the rise event error covariance to make
    it positive definite.

    This function does not verify that the computed covariance matrix is
    positive definite.  However, it does verify that the matrix is symmetric,
    and therefore successful Cholesky decomposition of the matrix
    (numpy.linalg.cholesky) will confirm that it is positive definite.  See
    Press et al. (2021) Numerical Recipes, 3rd ed.

    """
    # XXX Partly duplicates code in find_offsets

    # Assemble mapping of series ids to row numbers (unknowns) for the offset-
    #   finding problem.  Note that these are not the same as indices in the
    #   series list, which must be obtained via index_mapping:
    #   a_series = series[index_mapping[series_id]]
    series_ids = sorted(
        set().union(
            *(
                (series_id for series_id, _ in seq)
                for seq in list(head_mapping.values())
            )
        )
    )

    check_rise_head_mapping(head_mapping, series, index_mapping, head_step)

    number_of_equations = sum(
        len(series_at_head) for series_at_head in list(head_mapping.values())
    )
    if not recharge_error_weight:
        return np.identity(number_of_equations)
    number_of_unknowns = len(series_ids) - 1
    # XXX /Duplicate code
    # (i, j)(k)
    ij = [
        (head_id, series_id)
        for head_id, series_at_head in sorted(head_mapping.items())
        for series_id, _ in series_at_head
    ]
    assert len(ij) == number_of_equations
    # k[i, j]
    eqn_no = {
        (head_id, series_id): eqn
        for eqn, (head_id, series_id) in enumerate(ij)
    }
    assert max(eqn_no.values()) == number_of_equations - 1

    omega = np.zeros(
        (number_of_equations, number_of_equations), dtype=np.float64
    )

    # R[j]
    rain_depth = {
        # Retrieving rain depth: obtain index, retrieve series;
        #   total_depth is the 2nd element of the first tuple
        series_id: series[index_mapping[series_id]][0][1]
        for series_id in series_ids
    }

    mean_rain_depth = sum(value for value in rain_depth.values()) / len(
        rain_depth
    )
    normalized_rain_depth = {
        key: value / mean_rain_depth for key, value in rain_depth.items()
    }
    del rain_depth
    # zeta_j^* - zeta_j^o
    rise = {
        # Retrieving rise: obtain index, retrieve series;
        #   initial_zeta, final_zeta is the second tuple
        series_id: (
            series[index_mapping[series_id]][1][1]
            - series[index_mapping[series_id]][1][0]
        )
        for series_id in series_ids
    }

    # zeta_bar
    mean_zeta = {
        series_id: sum(series[index_mapping[series_id]][1]) / 2
        for series_id in series_ids
    }

    # f[i, j]
    coef = {
        (head_id, series_id): (head_id * head_step - mean_zeta[series_id])
        / rise[series_id]
        for head_id, series_at_head in head_mapping.items()
        for series_id, _ in series_at_head
    }
    assert min(coef.values()) >= -0.5, min(coef.values())
    assert max(coef.values()) <= 0.5, max(coef.values())

    # To compute the terms in the covariance matrix, we need sets of heads
    # crossed by each series
    # I[j]
    series_heads = {}
    for head_id, series_at_head in head_mapping.items():
        for sid, _ in series_at_head:
            series_heads.setdefault(sid, []).append(head_id)
    # We also need sets of series crossing each head
    # J[i]
    Ji = {}
    for head_id, series_at_head in head_mapping.items():
        for sid, _ in series_at_head:
            Ji.setdefault(head_id, []).append(sid)

    # Terms
    # Rff[(j, i1, i2)] = R[j]^2 f[i_1, j] f[i_2, j]
    Rff = {}
    for series_id, head_ids in series_heads.items():
        for head_id_1 in head_ids:
            for head_id_2 in head_ids:
                Rff[(series_id, head_id_1, head_id_2)] = (
                    normalized_rain_depth[series_id] ** 2
                    * coef[(head_id_1, series_id)]
                    * coef[(head_id_2, series_id)]
                )

    # Populate covariance matrix
    for k1 in range(number_of_equations):
        i1, j1 = ij[k1]
        Ji1 = set(Ji[i1])
        for k2 in range(number_of_equations):
            i2, j2 = ij[k2]
            Ji2 = set(Ji[i2])
            if j1 == j2:
                omega[k1, k2] += Rff[(j1, i1, i2)]
            if j1 in Ji2:
                omega[k1, k2] -= Rff[(j1, i1, i2)] / len(Ji2)
            if j2 in Ji1:
                omega[k1, k2] -= Rff[(j2, i2, i1)] / len(Ji1)
            omega[k1, k2] += sum(
                Rff[(j, i1, i2)] for j in Ji1.intersection(Ji2)
            ) / (len(Ji1) * len(Ji2))

    # Basic checks
    assert omega.shape == (
        number_of_equations,
        number_of_equations,
    ), f'Covariance matrix shape {omega.shape} != (# eqs, #eqs)'
    assert np.isfinite(omega).all(), (
        'Covariance matrix contains non-finite values: '
        f'{omega[~np.isfinite(omega)]}'
    )
    # Matrix may not be exactly symmetric due to roundoff
    assert np.allclose(
        omega, omega.T
    ), 'Covariance matrix not nearly symmetric'
    # Make exactly symmetric
    i_lower = np.tril_indices(number_of_equations, -1)
    omega[i_lower] = omega.T[i_lower]
    assert (omega == omega.T).all(), 'Covariance matrix not symmetric'
    omega += np.identity(number_of_equations) / recharge_error_weight
    # Check that omega is positive definite
    L = np.linalg.cholesky(omega)
    assert np.allclose(np.dot(L, L.T), omega)
    return omega


def check_rise_head_mapping(head_mapping, series, index_mapping, head_step):
    """Check head mapping for rise curve assembly

    Verify that each rise contains exactly two depths that span the associated
    water level.
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
            assert (
                rise[0] <= zeta <= rise[1]
            ), f'Rise {rise} does not span {zeta}'
            del depth, rise, series_id, depth_at_head
        del head_id, zeta, series_at_head


def compute_rise_offsets(cursor, reference_zeta_mm, recharge_error_weight=0):
    """Compute storage offsets to populate rising_interval_zeta

    If a reference zeta is not None, the crossing-depth of this water level is
    used as the origin of the axis.  Otherwise, the highest water level in the
    longest assembled rise is used.

    """
    series, epoch, zeta_intervals = assemble_rise_series(cursor)
    delta_z_mm = get_head_step(cursor)
    # Solve for offsets
    indices, offsets, zeta_mapping = get_series_storage_offsets(
        series, delta_z_mm, recharge_error_weight=recharge_error_weight
    )

    reference_zeta_off_grid = (
        reference_zeta_mm is not None
        and not np.allclose(reference_zeta_mm % delta_z_mm, 0)
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
                'rain_depth_offset_mm': (
                    offsets[i] - mean_zero_crossing_depth_mm
                ),
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
            "Discrete water level interval not yet set"
        )
    return delta_z_mm


def assemble_rise_series(cursor):
    """Assemble rise series

    Returns:
      series:  A list of pairs of arrays
               ((0, total_depth), (initial_zeta, final_zeta)),
               each representing a distinct rise event.
      epoch:  Time vector as seconds since the UNIX epoch
      zeta_intervals:  A list of (start, thru) indices in the epoch vector,
                       one for each series

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
        # Epoch associated with rainfall intensities are *start*
        # epoch for the interval, so the time slice that *starts*
        # at the storm thru_epoch is not included.
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
        assert (
            len(zeta_seq) >= 2
        ), 'A jump is defined by at least two zeta values'
        assert (
            np.diff(zeta_seq) > 0
        ).all(), f'{zeta_seq} is not strictly increasing'
        initial_zeta = zeta_seq[0]
        final_zeta = zeta_seq[-1]
        assert (
            len(zeta_seq) > 0
        ), 'empty sequence'  # pylint:disable=len-as-condition
        rain_intervals.append((rain_start, rain_stop))
        zeta_intervals.append((zeta_start, zeta_thru + 1))
        series.append(
            (np.array((0, total_depth)), np.array((initial_zeta, final_zeta)))
        )

    del zeta_mm
    return series, epoch, zeta_intervals
