# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Simulate scalar water table and peat surface dynamics"""

import logging

import numpy as np

import spowtd._cumulant as cumulant_mod
import spowtd.dynamic_0d._simulator_0d as simulator_mod


LOG = logging.getLogger('spowtd.dynamic_0d.dynamic_solve')


def solve_0d_surface(solver, time_grid, initial_conditions):
    """Simulate scalar water table and peat surface dynamics

    * time_grid is the vector of times at which to store output
    * initial_conditions gives the water level and surface at time_grid[0]

    Returns the state trajectory as a dict with these keys:
    - surface_mm
    - water_level_mm
    - elapsed_time_d

    """
    trajectory = np.empty((len(time_grid), 2))
    trajectory[:] = float('NaN')
    # zeta, p
    trajectory[0, :] = list(initial_conditions)
    try:
        time_grid = np.array(time_grid, dtype=np.float64)
    except TypeError:
        raise ValueError(
            'Time grid not coercible to float array: {}'.format(time_grid)
        ) from None
    solver.simulate(time_grid, trajectory)
    water_level, surface = (  # pylint: disable=unbalanced-tuple-unpacking
        trajectory.T.tolist()
    )
    trajectory = {
        'surface_mm': surface,
        'water_level_mm': water_level,
        'elapsed_time_d': time_grid.tolist(),
    }
    return trajectory


def instantiate_solver(
    solver_name,
    numerical_parameters,
    peat_properties,
    recharge,
    curvature,
    solver_trajectory_file=None,
):
    """Instantiate dynamic 0d solver

    * solver_name is the name of the solver to use
    * numerical_parameters is a dict of parameters for the solver
    * recharge is a time series
    * curvature is the negative Laplacian of the peat surface elevation

    """
    solver_classes = {
        'fixed-step-water_table': simulator_mod.FixedStepWaterTableSurfaceSimulator0D,
        'fixed-step-storage': simulator_mod.FixedStepStorageSurfaceSimulator0D,
    }
    if solver_name not in solver_classes:
        raise ValueError(
            'Unrecognized solver "{}"; choose from: {}'.format(
                solver_name, list(sorted(solver_classes.keys()))
            )
        )
    solver_class = solver_classes[solver_name]
    LOG.info('Using simulator %s with %s', solver_class.__name__, numerical_parameters)
    return solver_class(
        solver_class.state_class(
            transmissivity=peat_properties.transmissivity,
            peat_growth=peat_properties.peat_growth,
            specific_yield=peat_properties.specific_yield,
            recharge=recharge,
            laplacian=-curvature,
            solver_trajectory_file=solver_trajectory_file,
        ),
        **numerical_parameters,
    )


def postprocess_trajectory(
    transmissivity, specific_yield, curvature, recharge, trajectory
):
    """Add auxiliary variables to trajectory

    The added variables are:
    - recharge_mm_d
    - transmissivity_m2_d
    - storage_mm
    - net_recharge_mm (scalar cumulant)

    Mutates the trajectory, returning None

    """
    time_grid = trajectory['elapsed_time_d']
    trajectory['recharge_mm_d'] = [recharge(t) for t in time_grid]
    trajectory['transmissivity_m2_d'] = [
        transmissivity(zeta) for zeta in trajectory['water_level_mm']
    ]
    trajectory['storage_mm'] = [
        specific_yield.shallow_storage(zeta) for zeta in trajectory['water_level_mm']
    ]
    zeta = trajectory['water_level_mm']
    # Use integral for recharge, trapezoid rule for discharge
    trajectory['net_recharge_mm'] = full_precision_sum(
        recharge.integral(time_grid[i], time_grid[i + 1])
        - (time_grid[i + 1] - time_grid[i])
        * curvature
        * (transmissivity(zeta[i]) / 2 + transmissivity(zeta[i + 1]) / 2)
        for i in range(len(time_grid) - 1)
    )


def full_precision_sum(iterable):
    """Compute the full-precision sum of an iterable"""
    fp_op = cumulant_mod.Operator()
    return fp_op.sum(np.array(list(iterable), dtype='float64'))
