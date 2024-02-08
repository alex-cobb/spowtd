"""Test code for rise module

"""

import os

import numpy as np

import pytest

import spowtd.rise as rise_mod
import spowtd.test


omega_expected = {2: []}


def test_rise(classified_connection, request):
    """Assemble rising curve"""
    rise_mod.find_rise_offsets(classified_connection)


def test_rise_covariance(classified_connection, request):
    """Assemble covariance of rise event errors"""
    sample_id = spowtd.test.get_dataset_id(request)
    if sample_id != 2:
        pytest.skip(f'No test data for dataset {sample_id}')
    omega = rise_mod.get_rise_covariance(
        classified_connection, recharge_error_weight=1e3
    )
    with open(
        os.path.join(os.path.dirname(__file__), 'sample_data', 'omega_2.txt'),
        'rt',
        encoding='utf-8',
    ) as f:
        omega_ref = np.loadtxt(f)
    assert np.allclose(omega_ref, omega)


def test_rise_with_covariance(classified_connection):
    """Assemble rise curve considering covariance from storm depth errors"""
    rise_mod.find_rise_offsets(
        classified_connection, recharge_error_weight=1e3
    )
