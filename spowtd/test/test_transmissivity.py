"""Test code for transmissivity classes

"""

import numpy as np

import pytest

import yaml

import spowtd.transmissivity as transmissivity_mod
from spowtd.test import conftest


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
    with open(conftest.get_parameter_file_path(T_type), 'rt') as T_file:
        T_parameters = yaml.safe_load(T_file)['transmissivity']
    transmissivity = transmissivity_mod.create_transmissivity_function(
        T_parameters
    )
    if T_type == 'spline':
        zeta_mm = np.linspace(-0.35, 0.2, 10) * 1000
        assert np.allclose(transmissivity(zeta_mm), expected_T)
    else:
        assert T_type == 'peatclsm'
        # In this case, expected_T is provided by an R script, and is
        # large (201 values), so we load it from a file produced by
        # the R script instead of including it in the pytest
        # decorator.
        del expected_T
        T_table = conftest.peatclsm_transmissivity_table()
        zeta_m = np.linspace(-1.5, 0.0, 151)[::-1]
        zeta_m_ref, expected_T = zip(*T_table)
        assert np.allclose(zeta_m, zeta_m_ref)
        assert np.allclose(transmissivity(zeta_m * 1000), expected_T)
