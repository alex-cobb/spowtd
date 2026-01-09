# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for simulating recession curves"""

import numpy as np

import pytest

import yaml

import spowtd.recession as recession_mod
import spowtd.simulate_recession as simulate_recession_mod
import spowtd.functions.specific_yield as specific_yield_mod
import spowtd.set_curvature as set_curvature_mod
import spowtd.functions.transmissivity as transmissivity_mod
from spowtd.test import conftest
from spowtd.test.utils import assert_close


@pytest.mark.parametrize('parameterization_type', ['peatclsm', 'spline'])
def test_simulate_recession_curve(classified_connection, parameterization_type):
    """Test simulation of recession curve"""
    # XXX Use a fixture
    recession_mod.find_recession_offsets(classified_connection)

    curvature_m_km2 = 2.36
    et_mm_d = 4.15
    mean_elapsed_time_d = 19.0

    set_curvature_mod.set_curvature(
        classified_connection, curvature_m_km2=curvature_m_km2
    )

    cursor = classified_connection.cursor()
    cursor.execute(
        """
    SELECT count(distinct zeta_number)
    FROM recession_interval_zeta"""
    )
    n_zeta = cursor.fetchone()[0]
    cursor.close()

    expected_curve = {
        ('peatclsm', 388): [
            0.54318592,
            6.22354904,
            11.04775281,
            15.14873985,
            18.69618036,
            21.86714916,
            24.82033685,
            27.68090068,
            30.53555132,
            33.436654,
        ],
        ('spline', 388): [
            10.24100329,
            12.79745897,
            14.87692191,
            16.74909081,
            18.5564347,
            20.30350164,
            21.92615134,
            23.4015907,
            24.84981245,
            26.2980342,
        ],
        ('peatclsm', 287): [
            0.54318592,
            6.22354904,
            11.04775281,
            15.14873985,
            18.69618036,
            21.86714916,
            24.82033685,
            27.68090068,
            30.53555132,
            33.436654,
        ],
        ('spline', 287): [
            10.24100329,
            12.79745897,
            14.87692191,
            16.74909081,
            18.5564347,
            20.30350164,
            21.92615134,
            23.4015907,
            24.84981245,
            26.2980342,
        ],
    }[(parameterization_type, n_zeta)]

    with open(
        conftest.get_parameter_file_path(parameterization_type),
        'rt',
        encoding='utf-8',
    ) as parameter_file:
        parameters = yaml.safe_load(parameter_file)
    specific_yield = specific_yield_mod.create_specific_yield_function(
        **parameters['specific_yield']
    )
    transmissivity_m2_d = transmissivity_mod.create_transmissivity_function(
        **parameters['transmissivity']
    )
    # Transmissivity parameters have a ceiling at 1 cm
    zeta_grid_mm = np.linspace(0, -400, 10)
    recession_curve = simulate_recession_mod.compute_recession_curve(
        specific_yield,
        transmissivity_m2_d,
        zeta_grid_mm,
        mean_elapsed_time_d=mean_elapsed_time_d,
        curvature_km=curvature_m_km2 * 1e-3,
        et_mm_d=et_mm_d,
    )
    assert_close(recession_curve.mean(), mean_elapsed_time_d)
    assert_close(recession_curve, expected_curve)
