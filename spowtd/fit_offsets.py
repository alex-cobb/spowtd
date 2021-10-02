"""Determine time offsets that align pieces of a hydrograph

"""

import logging

import numpy.linalg as linalg_mod
import numpy as np

import spowtd.regrid as regrid_mod


LOG = logging.getLogger('spowtd.fit_offsets')


def get_series_time_offsets(series_list, head_step):
    """Find a time offset that minimizes difference in head crossing times

    Given a sequence of (time, head) data series, rediscretize to get
    an average crossing time of each integer multiple of head_step for
    each series.  Then find a time offset for each data series that
    minimizes the sum of squared differences in times when all data
    series cross those head values.  This offset *replaces* the existing
    offset of the series, t_new = t - min(t) + time_offset

    The series in the list with the largest initial head is treated as
    the reference series, and assigned a time offset of zero.

    Returns (indices, time_offsets, head_mapping) connecting indices
    of series in series_list to their corresponding time offset.  In
    general, these will be a subset of series_list if there were some
    heads at which there was no overlap.  The head mapping is a
    mapping between discrete head ids and sequences of (series_id,
    t_mean) pairs, representing the mean time at which that series
    crossed that head.  Head ids are integers which, when multiplied
    by head_step, will give once more a head with units.

    """
    if not series_list:
        raise ValueError('empty series list')
    # We need to retain the indices in series_list so that the caller
    # knows which offsets go with which series, but we also need to
    # sort by initial head; so, retain a mapping from the series_id
    # we use for finding offsets to index in sorted_list
    dec = sorted(((t - t.min(), H, index) for index, (t, H)
                  in enumerate(series_list)),
                 key=lambda t_H_index: t_H_index[1][0])
    sorted_list = []
    index_mapping = {}
    for new_index, (t, H, original_index) in enumerate(dec):
        sorted_list.append((t, H))
        index_mapping[new_index] = original_index
    del dec
    head_mapping = build_head_mapping(sorted_list, head_step)
    series_at_head = dict((head, set(series_id for
                                     series_id, t_mean in value))
                          for head, value
                          in list(head_mapping.items()))
    connected_components = get_connected_components(series_at_head)
    if len(connected_components) > 1:
        LOG.info(
            '{} connected sets of head of sizes '
            '{}; will keep only largest component.'
            .format(len(connected_components),
                    tuple(len(cc) for cc in
                          connected_components)))

    head_mappings = split_mapping_by_keys(head_mapping,
                                          connected_components[:1])
    assert len(head_mappings) == 1
    head_mapping = head_mappings[0]
    del head_mappings

    series_ids, offsets = find_offsets(head_mapping)
    assert len(series_ids) == len(offsets)
    original_indices = [index_mapping[series_id] for series_id in series_ids]
    # we also need to map series ids in head_mapping to their original
    # indices
    output_mapping = {}
    for head_id, crossings in list(head_mapping.items()):
        output_mapping[head_id] = [(index_mapping[series_id],
                                    t_mean) for series_id, t_mean
                                   in crossings]
    return (original_indices, offsets, output_mapping)


def build_head_mapping(series, head_step=1):
    """Construct a mapping between head ids and series crossing times

    series is a sequence of (time, head) data series.  Each series is
    regridded via interpolation to instead give times at which the
    head time series crosses an integer multiple of head_step.  That
    integer multiple (head id) is the key to the returned mapping,
    which maps between head_id and a sequence of (series_id, time)
    pairs indicating when each series crossed that head.  The series
    id is just the index of the series passed in.
    """
    head_mapping = {}
    for series_id, (t, H) in enumerate(series):
        # take averages of t at distinct H
        all_times = {}
        for head_id, t in regrid_mod.regrid(t, H, head_step):
            all_times.setdefault(head_id, []).append(t)
        for head_id, t in list(all_times.items()):
            t_mean = np.mean(t)
            head_mapping.setdefault(head_id, []).append((series_id,
                                                         t_mean))
    return head_mapping


def find_offsets(head_mapping):
    """Find the time offsets that align the series in head_mapping

    Finds the set of time offsets that minimize the sum of squared
    differences in times at which each series crosses a particular
    head.  Input is a mapping of head id (a hashable value
    corresponding to a head, normally an integer) to a sequence of
    (series_id, time) pairs wherein series_id is an identifier for a
    sequence and time is the time at which the series crossed the
    corresponding head value.

    The series with the series_id that is largest (last in sort order)
    is treated as the reference and given an offset of zero; all other
    offsets are relative to that one.

    Returns series_ids, offsets where series_ids are the identifiers

    """
    # Eliminate all heads with only one series, these are
    #   uninformative
    for head_id, seq in list(head_mapping.items()):
        # Don't use "assert seq" here, this is an ndarray
        assert len(seq) > 0  # pylint: disable=len-as-condition
        if len(seq) == 1:
            del head_mapping[head_id]
    # Assemble mapping of series ids to row numbers for the
    # least-squares problem
    series_ids = ((series_id for series_id, t_mean in seq)
                  for seq in list(head_mapping.values()))
    series_ids = sorted(set().union(*series_ids))
    series_indices = dict(zip(series_ids,
                              range(len(series_ids))))
    # Reference series corresponds to the highest series id; it
    #   has the largest initial head, because we sorted them
    reference_index = max(series_ids)
    LOG.info('Reference index: {}'.format(reference_index))
    number_of_equations = sum(len(series_at_head) for series_at_head
                              in list(head_mapping.values()))
    number_of_unknowns = len(series_indices) - 1
    LOG.info('{} equations, {} unknowns'.format(number_of_equations,
                                                number_of_unknowns))
    A = np.zeros((number_of_equations, number_of_unknowns))
    b = np.zeros((number_of_equations,))
    row_template = np.zeros((number_of_unknowns,))
    row_index = 0
    for head_id, series_at_head in list(head_mapping.items()):
        row_template[:] = 0
        sids, times = list(zip(*series_at_head))
        number_of_series_at_head = len(sids)
        indices = [series_indices[index] for index in sids
                   if index != reference_index]
        row_template[indices] = 1. / number_of_series_at_head
        mean_time = np.mean(times)
        for series_id, t in series_at_head:
            A[row_index] = row_template
            # !!! some redundancy here
            if series_id != reference_index:
                series_index = series_indices[series_id]
                A[row_index, series_index] -= 1
            b[row_index] = t - mean_time
            row_index += 1
    assert row_index == number_of_equations, row_index
    ATA = np.dot(A.transpose(), A)
    assert ATA.shape == (number_of_unknowns,
                         number_of_unknowns), ATA.shape
    ATd = np.dot(A.transpose(), b)
    offsets = linalg_mod.solve(ATA, ATd)  # pylint: disable=E1101
    # this was the boundary condition, zero offset for
    # reference (last) id
    offsets = np.concatenate((offsets, [0]))
    # offsets are by index, but reverse mapping is trivial
    # because series ids are sorted
    assert len(series_ids) == len(offsets), \
        '{} != {}'.format(len(series_ids), len(offsets))
    return (series_ids, offsets)


def split_mapping_by_keys(mapping, key_lists):
    """Split up a mapping (dict) according to connected components

    Each connected component is a sequence of keys from mapping;
    returned is a corresponding sequence of head mappings, each with
    only the keys from that connected component.
    """
    mappings = []
    for seq in key_lists:
        mappings.append(dict((head_id, value) for head_id, value
                             in list(mapping.items()) if head_id in seq))
    return mappings


def get_connected_components(head_mapping):
    """Find all overlapping sequences of heads in head_mapping

    Head_mapping is a mapping between head_ids and sets of ids for
    series that have data at that head_id.  Connected components are
    tuples of series ids that overlap in head.

    Series id sequences are returned sorted from longest to shortest.

    Head ids rather than heads are used because they must be hashable.

    """
    groups = {}
    for head_id, series_at_head in list(head_mapping.items()):
        matches = [keys for keys, group in list(groups.items())
                   if not series_at_head.isdisjoint(group)]
        new_keys = (head_id,) + sum(matches, ())
        others = [groups.pop(keys) for keys in matches]
        new_group = series_at_head.union(*others)
        groups[new_keys] = new_group
    connected_components = sorted(list(groups.keys()),
                                  key=len, reverse=True)
    # sanity check: union should include all head_id ids
    assert (sum(len(cc) for cc in connected_components) ==
            len(head_mapping))
    assert (set().union(*[set(cc) for cc in connected_components]) ==
            set(head_mapping))
    return connected_components
