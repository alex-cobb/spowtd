# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for transmissivity classes"""

import numpy as np

import pytest

import yaml

import spowtd.functions.transmissivity as transmissivity_mod
from spowtd.test import conftest
from spowtd.test.utils import assert_close


@pytest.mark.parametrize(
    'T_type, expected_T',
    [
        ('peatclsm', None),  # Expected values retrieved from R script
        (
            'spline',
            [
                7.44200000e00,
                7.45744939e00,
                8.09108022e00,
                1.00248694e01,
                1.59266347e01,
                3.39383370e01,
                1.02036399e02,
                1.36457534e03,
                2.92901037e04,
                3.39326681e05,
            ],
        ),
    ],
)
def test_transmissivity(T_type, expected_T):
    """Test transmissivity functions"""
    with open(
        conftest.get_parameter_file_path(T_type),
        'rt',
        encoding='utf-8',
    ) as T_file:
        T_parameters = yaml.safe_load(T_file)['transmissivity']
    transmissivity = transmissivity_mod.create_transmissivity_function(**T_parameters)
    if T_type == 'spline':
        zeta_mm = np.linspace(-0.35, 0.2, 10) * 1000
        assert_close(transmissivity(zeta_mm), expected_T)
        assert_close(np.vectorize(transmissivity._T)(zeta_mm), expected_T)
    else:
        assert T_type == 'peatclsm'
        # In this case, expected_T is provided by an R script, and is large (201
        # values), so we load it from a file produced by the R script instead of
        # including it in the pytest decorator.
        del expected_T
        try:
            T_table = conftest.peatclsm_transmissivity_table()
        except FileNotFoundError:
            pytest.skip('Rscript not found')
        zeta_m = np.linspace(-1.5, 0.0, 151)[::-1]
        zeta_m_ref, expected_T = zip(*T_table)
        assert_close(zeta_m, zeta_m_ref)
        assert_close(transmissivity(zeta_m * 1000), expected_T)
