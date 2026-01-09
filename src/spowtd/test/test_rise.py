# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for rise module"""

import os

import numpy as np

import spowtd.rise as rise_mod
import spowtd.test
from spowtd.test.utils import assert_close


omega_expected = {2: []}


def test_rise(classified_connection):
    """Assemble rising curve"""
    rise_mod.find_rise_offsets(classified_connection)


def test_rise_covariance(classified_connection, request):
    """Assemble covariance of rise event errors"""
    sample_id = spowtd.test.get_dataset_id(request)
    omega = rise_mod.get_rise_covariance(
        classified_connection, recharge_error_weight=1e3
    )
    # Test for regression
    with open(
        os.path.join(
            os.path.dirname(__file__), 'sample_data', f'omega_{sample_id}.txt'
        ),
        'rt',
        encoding='utf-8',
    ) as f:
        omega_ref = np.loadtxt(f)
    assert_close(omega_ref, omega)


def test_rise_with_covariance(classified_connection):
    """Assemble rise curve considering covariance from storm depth errors"""
    # Smoke test
    rise_mod.find_rise_offsets(classified_connection, recharge_error_weight=1e3)
