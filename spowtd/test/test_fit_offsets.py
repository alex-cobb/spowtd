"""Test code for fit_offsets module

"""

import numpy as np

import pytest

import spowtd.fit_offsets as fo_mod


head_mapping_test_cases = {
    # Uniform R = 4
    # 5 equations, 1 unknown
    1: {1: [(1, -2), (2, -2)], 2: [(1, 2), (2, 0)], 3: [(2, 2)]},
    # Uniform R = 6
    # 7 equations, 2 unknowns.
    # In this case b (rhs) is the same for weighted and unweighted assembly.
    2: {
        1: [(1, -3), (2, -3)],
        2: [(1, 3), (2, 0), (3, -3)],
        3: [(2, 3), (3, 3)],
    },
    # Uniform R = 6
    # 6 equations, 1 unknown
    3: {1: [(1, -3), (2, -3)], 2: [(1, 3), (2, -1)], 3: [(2, 1)], 4: [(2, 3)]},
    # Nonuniform R: R_1 = 6, R_2 = 1
    # 6 equations, 1 unknown
    4: {1: [(1, -3)], 2: [(1, 0), (2, -1)], 3: [(1, 3), (2, 0)], 4: [(2, 1)]},
}

series_ids_expected = {1: [1, 2], 2: [1, 2, 3], 3: [1, 2], 4: [1, 2]}

# _u for unweighted
b_expected_u = {
    1: [0, 0, 1, -1, 0],
    2: [0, 0, 3, 0, -3, 0, 0],
    3: [0, 0, 2, -2, 0, 0],
    4: [0, 1 / 2, -1 / 2, 3 / 2, -3 / 2, 0],
}
b_expected = {
    1: [0, 0, 5 / 3, -1 / 3, 0],
    2: [0, 0, 3, 0, -3, 0, 0],
    3: [0, 0, 10 / 3, -2 / 3, 0, 0],
    4: [0, 1 / 3, -2 / 3, 30 / 11, -3 / 11, 0],
}

A_expected_u = {
    1: [[-1 / 2], [1 / 2], [-1 / 2], [1 / 2], [0]],
    2: [
        [-1 / 2, 1 / 2],
        [1 / 2, -1 / 2],
        [-2 / 3, 1 / 3],
        [1 / 3, -2 / 3],
        [1 / 3, 1 / 3],
        [0, -1 / 2],
        [0, 1 / 2],
    ],
    3: [[-1 / 2], [1 / 2], [-1 / 2], [1 / 2], [0], [0]],
    4: [[0], [-1 / 2], [1 / 2], [-1 / 2], [1 / 2], [0]],
}
A_expected = {
    1: [[-1 / 2], [1 / 2], [-5 / 6], [1 / 6], [0]],
    2: [
        [-1 / 2, 1 / 2],
        [1 / 2, -1 / 2],
        [-11 / 12, 5 / 6],
        [1 / 12, -1 / 6],
        [1 / 12, 5 / 6],
        [0, -1 / 2],
        [0, 1 / 2],
    ],
    3: [[-1 / 2], [1 / 2], [-5 / 6], [1 / 6], [0], [0]],
    4: [[0], [-1 / 3], [2 / 3], [-10 / 11], [1 / 11], [0]],
}

offsets_expected_u = {1: [-1, 0], 2: [-4, -2, 0], 3: [-2, 0], 4: [-2, 0]}
offsets_expected = {}


@pytest.mark.parametrize("test_case", [1, 2, 3, 4])
def test_assemble_linear_system_unweighted(test_case):
    """Unweighted test cases for linear system assembly"""
    head_mapping = head_mapping_test_cases[test_case]
    series_ids = series_ids_expected[test_case]
    series_indices = dict(zip(series_ids, range(len(series_ids))))
    A, b = fo_mod.assemble_linear_system(head_mapping, series_indices)
    A_ref = np.array(A_expected_u[test_case], dtype=float)
    b_ref = np.array(b_expected_u[test_case], dtype=float)
    assert b.shape == b_ref.shape
    assert np.allclose(b_ref, b)
    assert A.shape == A_ref.shape
    for i, ref in enumerate(A_ref):
        assert np.allclose(ref, A[i]), i


@pytest.mark.parametrize("test_case", [1, 2, 3, 4])
def test_assemble_weighted_linear_system(test_case):
    """Weighted test cases for linear system assembly"""
    head_mapping = head_mapping_test_cases[test_case]
    series_ids = series_ids_expected[test_case]
    series_indices = dict(zip(series_ids, range(len(series_ids))))
    A, b = fo_mod.assemble_weighted_linear_system(
        head_mapping, series_indices, recharge_error_weight=1
    )
    A_ref = np.array(A_expected[test_case], dtype=float)
    b_ref = np.array(b_expected[test_case], dtype=float)
    assert b.shape == b_ref.shape
    assert np.allclose(b_ref, b)
    assert A.shape == A_ref.shape
    for i, ref in enumerate(A_ref):
        assert np.allclose(ref, A[i]), i


@pytest.mark.parametrize("test_case", [1, 2, 3, 4])
def test_find_offsets_unweighted(test_case):
    """Unweighted test cases for find_offsets"""
    head_mapping = head_mapping_test_cases[test_case]
    series_ids, offsets = fo_mod.find_offsets(head_mapping, covariance=None)
    assert series_ids == series_ids_expected[test_case]
    assert isinstance(offsets, np.ndarray)
    assert offsets.shape == (len(series_ids),)
    assert np.allclose(offsets, offsets_expected_u[test_case])


@pytest.mark.parametrize("test_case", [1])
def test_find_offsets_weighted(test_case):
    """Weighted test cases for find_offsets"""
    pytest.skip('Not implemented yet: finding offsets with covariance matrix')
    head_mapping = head_mapping_test_cases[test_case]
    series_ids, offsets = fo_mod.find_offsets(head_mapping, covariance=None)
    assert series_ids == series_ids_expected[test_case]
    assert isinstance(offsets, np.ndarray)
    assert offsets.shape == (len(series_ids),)
    assert np.allclose(offsets, offsets_expected_u[test_case])
