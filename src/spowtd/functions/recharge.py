# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Scalar recharge functions for simulations"""

import logging

import numpy as np

import spowtd._scalar_function as scalar_function_mod


LOG = logging.getLogger('spowtd.functions.recharge')


def create_recharge_function(recharge_parameters):
    """Create scalar recharge function"""
    function_type = recharge_parameters.pop('type')

    factories = {
        'constant': create_constant_recharge,
        'periodic_storm': create_periodic_storm_recharge,
        'piecewise_constant': create_piecewise_constant_recharge,
    }

    try:
        factory = factories[function_type]
    except KeyError:
        raise ValueError(
            'Unrecognized recharge function type {}'.format(function_type)
        ) from None
    return factory(**recharge_parameters)


def create_constant_recharge(precipitation_mm_d, evapotranspiration_mm_d):
    """Create constant scalar recharge function"""
    return scalar_function_mod.ConstantScalarFunction(
        precipitation_mm_d - evapotranspiration_mm_d
    )


def create_periodic_storm_recharge(
    storm_depth_mm,
    storm_intensity_mm_h,
    precipitation_mm_d,
    evapotranspiration_mm_d,
    phase_d=0,
):
    """Create a scalar recharge function with periodic storms"""
    if evapotranspiration_mm_d < 0:
        LOG.warning(
            'Negative evapotranspiration %s mm / d (condensation)',
            evapotranspiration_mm_d,
        )
    box_height_mm_d = storm_intensity_mm_h * 24.0
    box_width_d = storm_depth_mm / box_height_mm_d
    assert np.allclose(storm_depth_mm, box_height_mm_d * box_width_d)
    mean_intensity_mm_d = precipitation_mm_d
    period_d = storm_depth_mm / mean_intensity_mm_d
    assert np.allclose(mean_intensity_mm_d, storm_depth_mm / period_d)
    if box_width_d > period_d:
        LOG.warning('Storm duration %s d exceeds period %s d', box_width_d, period_d)
    return scalar_function_mod.PeriodicBoxcarFunction(
        box_height=box_height_mm_d,
        box_width=box_width_d,
        period=period_d,
        phase=phase_d,
        baseline=-evapotranspiration_mm_d,
    )


def create_piecewise_constant_recharge(elapsed_time_d, recharge_mm_d):
    """Create piecewise-constant scalar recharge function"""
    return scalar_function_mod.PiecewiseConstantFunction(
        time=np.asarray(elapsed_time_d), value=np.asarray(recharge_mm_d)
    )
