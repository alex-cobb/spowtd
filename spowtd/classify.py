"""Classify data into storm and interstorm intervals

"""

import logging

import numpy as np


LOG = logging.getLogger('spowtd.classify')


def classify_intervals(
    connection, storm_rain_threshold_mm_h=4.0, rising_jump_threshold_mm_h=8.0
):
    """Classify data into storm and interstorm intervals"""
    cursor = connection.cursor()
    cursor.execute(
        """
    INSERT INTO thresholds
      (storm_rain_threshold_mm_h, rising_jump_threshold_mm_h)
    VALUES
      (:storm_rain_threshold_mm_h, :rising_jump_threshold_mm_h)""",
        {
            'storm_rain_threshold_mm_h': storm_rain_threshold_mm_h,
            'rising_jump_threshold_mm_h': rising_jump_threshold_mm_h,
        },
    )
    cursor.execute(
        """
    SELECT DISTINCT data_interval
    FROM grid_time
    WHERE data_interval IS NOT NULL
    ORDER BY data_interval"""
    )
    data_intervals = [row[0] for row in cursor.fetchall()]
    if not data_intervals:
        raise ValueError('No valid data intervals found')
    for data_interval in data_intervals:
        populate_zeta_interval(
            cursor,
            data_interval,
            storm_rain_threshold_mm_h,
            rising_jump_threshold_mm_h,
        )
    cursor.close()
    connection.commit()


def populate_zeta_interval(
    cursor,
    data_interval,
    storm_rain_threshold_mm_h,
    rising_jump_threshold_mm_h,
):
    """Identify storm and interstorm intervals"""
    classify_interstorms(cursor, data_interval, rising_jump_threshold_mm_h)
    match_all_storms(
        cursor,
        data_interval,
        storm_rain_threshold_mm_h,
        rising_jump_threshold_mm_h,
    )


def classify_interstorms(cursor, data_interval, rising_jump_threshold_mm_h):
    """Populate interstorm intervals"""
    (epoch, zeta_mm, is_raining) = (
        np.array(v)
        for v in zip(
            *cursor.execute(
                """
         SELECT water_level.epoch,
                zeta_mm,
                rainfall_intensity_mm_h > 0 AS is_raining
         FROM grid_time
         JOIN rainfall_intensity
           ON rainfall_intensity.from_epoch = grid_time.epoch
           AND grid_time.data_interval = ?
         JOIN water_level
           ON rainfall_intensity.from_epoch = water_level.epoch
         ORDER BY from_epoch""",
                (data_interval,),
            )
        )
    )
    assert len(epoch), epoch.shape
    check_for_uniform_time_steps(epoch)
    hour = epoch / 3600.0
    is_raining = is_raining.astype(bool)
    assert np.isfinite(zeta_mm).all()
    # Look for jumps in head much bigger than noise, which could
    # indicate the onset of rain, and mark everything after the jump
    # until the next rain as a "mystery jump".
    rates = np.concatenate(
        ([0], (zeta_mm[1:] - zeta_mm[:-1]) / (hour[1:] - hour[:-1]))
    )
    is_jump = (rates > rising_jump_threshold_mm_h).astype(bool)
    is_mystery_jump = get_mystery_jump_mask(is_jump, is_raining)
    is_interstorm = (~is_mystery_jump) & (~is_raining)
    interval_mask = is_interstorm
    del is_raining

    cursor.executemany(
        """
    INSERT INTO grid_time_flags
      (start_epoch, is_jump, is_mystery_jump, is_interstorm)
    VALUES
      (?, ?, ?, ?)""",
        zip(
            (int(t) for t in epoch),
            (int(b) for b in is_jump),
            (int(b) for b in is_mystery_jump),
            (int(b) for b in is_interstorm),
        ),
    )
    del is_jump, is_mystery_jump, is_interstorm

    masks = get_true_interval_masks(interval_mask)

    mask = None
    series_indices = [np.nonzero(mask)[0] for mask in masks]
    del mask, masks
    indices = None
    series_indices = [
        indices for indices in series_indices if len(indices) > 1
    ]
    del indices

    LOG.info('%s series found', len(series_indices))

    for indices in series_indices:
        cursor.execute(
            """
        INSERT INTO zeta_interval
          (start_epoch, interval_type, thru_epoch)
        SELECT
          :start_epoch, :interval_type, :thru_epoch""",
            {
                'interval_type': 'interstorm',
                'start_epoch': int(epoch[indices[0]]),
                'thru_epoch': int(epoch[indices[-1]]),
            },
        )
    del hour, zeta_mm
    del series_indices


def match_all_storms(
    cursor,
    data_interval,
    storm_rain_threshold_mm_h,
    rising_jump_threshold_mm_h,
):
    """Match intervals of increasing head with rainstorms

    Calls match_storms on rain data and head data from the database.

    Thresholds are given in mm / h.

    """
    (epoch, zeta_mm, rainfall_intensity_mm_h) = (
        np.array(v)
        for v in zip(
            *cursor.execute(
                """
         SELECT water_level.epoch,
                zeta_mm,
                rainfall_intensity_mm_h
         FROM grid_time
         JOIN rainfall_intensity
           ON rainfall_intensity.from_epoch = grid_time.epoch
           AND grid_time.data_interval = ?
         JOIN water_level
           ON rainfall_intensity.from_epoch = water_level.epoch
         ORDER BY from_epoch""",
                (data_interval,),
            )
        )
    )
    check_for_uniform_time_steps(epoch)
    (time_step_h,) = cursor.execute(
        """
    SELECT CAST(time_step_s AS double precision) / 3600.
    FROM time_grid"""
    ).fetchone()
    jump_delta_threshold = rising_jump_threshold_mm_h * time_step_h
    (rain_intervals, jump_intervals) = match_storms(
        rainfall_intensity_mm_h,
        zeta_mm,
        storm_rain_threshold_mm_h,
        jump_delta_threshold,
    )

    is_storm = rainfall_intensity_mm_h > storm_rain_threshold_mm_h
    for i, (rain_start, rain_stop) in enumerate(rain_intervals):
        assert is_storm[rain_start:rain_stop].all(), (
            'Storm interval [{}, {}] includes only raining time steps'
            # XXX Convert to datetime for error message?
            .format(epoch[rain_start], epoch[rain_stop - 1])
        )
        jump_start, jump_stop = jump_intervals[i]
        assert (
            np.diff(zeta_mm[jump_start:jump_stop]) > jump_delta_threshold
        ).all()

        # Times associated with rainfall intensities are *start* times
        # for the interval, so the time slice that *starts* at the
        # storm thru_epoch is not included; but the slice that goes
        # thru the thru_epoch is.
        storm_start_epoch = int(epoch[rain_start])
        storm_thru_epoch = int(epoch[rain_stop])
        # On the other hand, heads are instantaneous values, so the
        # epoch of the end of the jump interval is the one to use.
        jump_start_epoch = int(epoch[jump_start])
        jump_thru_epoch = int(epoch[jump_stop - 1])
        # Duplicates are possible if multiple jumps match
        # the same storm
        already_seen = cursor.execute(
            """
        SELECT EXISTS (
          SELECT 1 FROM storm
          WHERE start_epoch = :start_epoch
          AND thru_epoch = :thru_epoch
        )""",
            {'start_epoch': storm_start_epoch, 'thru_epoch': storm_thru_epoch},
        ).fetchone()[0]
        if not already_seen:
            cursor.execute(
                """
            INSERT INTO storm (start_epoch, thru_epoch)
            SELECT :start_epoch, :thru_epoch""",
                {
                    'start_epoch': storm_start_epoch,
                    'thru_epoch': storm_thru_epoch,
                },
            )
        cursor.execute(
            """
        INSERT INTO zeta_interval
          (start_epoch, interval_type, thru_epoch)
        VALUES
          (:start_epoch, :interval_type, :thru_epoch)""",
            {
                'interval_type': 'storm',
                'start_epoch': jump_start_epoch,
                'thru_epoch': jump_thru_epoch,
            },
        )
        cursor.execute(
            """
        INSERT INTO zeta_interval_storm
          (interval_start_epoch, interval_type, storm_start_epoch)
        VALUES
          (:interval_start_epoch, :interval_type, :storm_start_epoch)""",
            {
                'interval_type': 'storm',
                'interval_start_epoch': jump_start_epoch,
                'storm_start_epoch': storm_start_epoch,
            },
        )


def match_storms(rain, head, rain_threshold, jump_threshold):
    """Match intervals of increasing head with rainstorms

    For contiguous intervals of head increasing faster than
    jump_threshold, look for a contiguous period of rain above
    rain_threshold.  If there is no overlapping interval of rain, the
    head interval is discarded; if there is more than one overlapping
    interval of rain, the one with highest total intensity is chosen.

    Returns (rain_intervals, head_intervals), two sequences of equal
    length containing the matching contiguous intervals.

    """
    is_raining = rain > rain_threshold
    rain_masks = list(get_true_interval_masks(is_raining))
    LOG.info(
        '%s intervals of rain above threshold %s mm / h',
        len(rain_masks),
        rain_threshold,
    )
    # initialize to -1; this will indicate interstorms
    storm_indices = np.zeros(len(is_raining), np.int64) - 1
    rain_mask = None
    for i, rain_mask in enumerate(rain_masks):
        storm_indices[rain_mask] = i
    del rain_mask
    assert ((storm_indices != -1) == is_raining).all()

    # Semantics: is_jump is True on an interval if the head was higher
    # at the end of the interval than it was at the beginning.
    # In other words, in the sequence [0, 1, 2, 2],
    # is_jump == [1, 1, 0] because the head values are recorded
    # at the boundaries between time intervals:
    #              *  *
    #           *
    # head = *
    #        |  |  |  |
    #       t0 t1 t2 t3
    # and head increased on the intervals (t0, t1] and
    # (t1, t2], but not on the interval (t2, t3].
    # In this case the (start, stop) slice corresponding to
    # the jump interval is (0, 2), suggesting head increase
    # from rain on the time interval (t0, t2]
    head_increments = np.diff(head)
    is_jump = head_increments > jump_threshold
    jump_masks = list(get_true_interval_masks(is_jump))
    LOG.info(
        '%s intervals of water level increment above threshold %s mm',
        len(jump_masks),
        jump_threshold,
    )
    rain_intervals = []
    head_intervals = []
    for i, jump_mask in enumerate(jump_masks):
        # No head was recorded at the end of the last interval, so
        # jump mask provides no information about rain in last
        # interval.
        intersection = is_raining[:-1] & jump_mask
        matching_storms = set(storm_indices[np.nonzero(intersection)[0]])
        assert -1 not in matching_storms, matching_storms
        if len(matching_storms) == 0:  # pylint: disable=len-as-condition
            continue
        if len(matching_storms) == 1:
            storm_index = matching_storms.pop()
        elif len(matching_storms) > 1:
            LOG.info('multiple matching storms {}'.format(matching_storms))
            storms_by_size = sorted(
                matching_storms, key=lambda j: sum(rain[rain_masks[j]])
            )
            storm_index = storms_by_size[-1]
        rain_indices = np.nonzero(rain_masks[storm_index])[0]
        assert is_raining[
            rain_indices
        ].all(), 'Storm includes only raining time steps'
        rain_start = rain_indices[0]
        rain_stop = rain_indices[-1] + 1
        assert (
            rain_start == 0 or not is_raining[rain_start - 1]
        ), 'No heavy rain just before slice'
        assert not is_raining[rain_stop], 'No heavy rain at end of slice'
        jump_indices = np.nonzero(jump_mask)[0]
        jump_start = jump_indices[0]
        jump_stop = jump_indices[-1] + 2
        assert (
            np.diff(head[jump_start:jump_stop]) > jump_threshold
        ).all(), 'Head pair [{}, {}) meets jump threshold {} mm'.format(
            jump_start, jump_stop, jump_threshold
        )
        assert jump_start == 0 or (
            head[jump_start] - head[jump_start - 1] <= jump_threshold
        ), 'Jump <= {} starts at jump_start: {}, {}'.format(
            jump_threshold, jump_start, head[jump_start] - head[jump_start - 1]
        )
        assert jump_stop == len(head) or (
            head[jump_stop] - head[jump_stop - 1] <= jump_threshold
        ), 'Jump > {} ends at jump_stop: {}, {}'.format(
            jump_threshold, jump_stop, head[jump_stop] - head[jump_stop - 1]
        )

        # At a minimum, all intervals include 1 rainfall intensity
        # value and 2 head values (note that these correspond to the
        # same duration: one time slice.
        assert rain_stop - rain_start >= 1
        assert jump_stop - jump_start >= 2
        rain_intervals.append((rain_start, rain_stop))
        head_intervals.append((jump_start, jump_stop))
        LOG.info(
            'Rain %s cell(s) before jump',
            (head_intervals[-1][0] - rain_intervals[-1][0]),
        )
        LOG.info(
            '  and %s time step(s) shorter',
            (
                (rain_intervals[-1][1] - rain_intervals[-1][0])
                - (head_intervals[-1][1] - head_intervals[-1][0] - 1)
            ),
        )
    return (rain_intervals, head_intervals)


def check_for_uniform_time_steps(epoch):
    """Verify that time steps are uniform

    Checks that time steps in sorted array epoch are uniform,
    otherwise raising ValueError.

    Note that this may not work as expected for inexact datatypes
    (floats).

    """
    delta_t = np.diff(epoch)
    if delta_t.min() != delta_t.max():
        raise ValueError(
            'Nonuniform time steps in {}'.format(sorted(set(delta_t)))
        )


def get_mystery_jump_mask(is_jump, is_raining):
    """Flag everything from a mystery jump until the next rain

    A "mystery jump" is a jump in head with no rain.  The returned
    mask has the same length as is_jump and is_raining, and is marked
    True during intervals [jump with no rain, rain).  If it's raining,
    at the time of the jump, nothing happens; if not, all time points
    until (but not including) the next True value in is_raining are
    flagged.  All times with jumps and no rain are flagged; all times
    with rain and no jump are not flagged; and times with no rain and
    no jump will be flagged if and only if they occur between a
    "mystery jump" (jump with no rain) and a rain event.

    """
    assert_equal(len(is_jump), len(is_raining))
    mystery_jump_mask = np.zeros(len(is_jump), bool)
    in_mystery = True
    # pylint: disable=consider-using-enumerate
    for i in range(len(mystery_jump_mask)):
        if is_raining[i]:
            in_mystery = False
        else:
            if is_jump[i]:
                in_mystery = True
        mystery_jump_mask[i] = in_mystery
    # Test assertions in docstring: mystery_jump_mask is true at least
    # wherever it's not raining and there's a jump, and false wherever
    # it's raining
    assert all(~mystery_jump_mask[np.nonzero(is_raining)[0]])
    assert all(mystery_jump_mask[np.nonzero(~is_raining & is_jump)[0]])
    return mystery_jump_mask


def get_true_interval_masks(boolean_vector):
    """Find a series of contigous intervals where boolean_vector is true

    Returns an iterator of masks for contiguous sets of indices where
    boolean_vector is true.  Each mask has the same length as boolean_vector.

    """
    if not boolean_vector.dtype == bool:
        raise ValueError('non-boolean input vector')
    int_vector = boolean_vector.astype(np.int64)
    is_start = np.concatenate(([0], int_vector[1:] - int_vector[:-1])) > 0
    # Indices increment for each block of True values in input vector
    indices = np.cumsum(is_start.astype(np.int64))
    indices[~boolean_vector] = 0
    assert (
        indices.astype(bool) == boolean_vector
    ).all(), 'Non-zero indices correspond to True elements of input vector'
    unique_indices = sorted(set(indices))
    assert unique_indices[0] == 0
    del unique_indices[0]
    return (indices == index for index in unique_indices)


def assert_equal(a, b):  # pylint:disable=invalid-name
    """Convenience function for checking equality"""
    assert a == b, '{} != {}'.format(a, b)
