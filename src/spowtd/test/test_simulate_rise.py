# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for simulating rise curves"""

import numpy as np

import pytest

import yaml

import spowtd.simulate_rise as simulate_rise_mod
import spowtd.specific_yield as specific_yield_mod
from spowtd.test import conftest


@pytest.mark.parametrize(
    'sy_type, expected_curve',
    [
        (
            'peatclsm',
            [
                -137.02531183,
                -103.31683955,
                -70.68835402,
                -39.36074849,
                -9.61918877,
                18.38953548,
                45.56722828,
                75.67822728,
                116.09909334,
                174.27635828,
            ],
        ),
        (
            'spline',
            [
                -58.42252048,
                -44.61618714,
                -30.80985381,
                -17.00352048,
                -3.19718714,
                10.60914619,
                24.84092401,
                41.60896966,
                60.02310417,
                86.96712502,
            ],
        ),
    ],
)
def test_compute_rise_curve(sy_type, expected_curve):
    """Test computation of rise curve"""
    with open(
        conftest.get_parameter_file_path(sy_type), 'rt', encoding='utf-8'
    ) as sy_file:
        sy_parameters = yaml.safe_load(sy_file)['specific_yield']
    specific_yield = specific_yield_mod.create_specific_yield_function(sy_parameters)
    zeta_grid_mm = np.linspace(-865, 50, 10)
    mean_storage_mm = 7.0
    rise_curve = simulate_rise_mod.compute_rise_curve(
        specific_yield, zeta_grid_mm, mean_storage_mm
    )
    assert np.allclose(rise_curve.mean(), mean_storage_mm)
    assert np.allclose(rise_curve, expected_curve)
