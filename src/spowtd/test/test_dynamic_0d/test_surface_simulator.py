# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for peat and hydrology time ODE"""

import warnings

import numpy as np

import scipy.integrate

import pytest

import spowtd.dynamic_0d._simulator_0d as sim_mod
import spowtd.dynamic_0d._state_0d as state_mod

from spowtd._scalar_function import ConstantScalarFunction, PythonScalarFunction
from spowtd.functions._transmissivity import TransmissivityNearSurface
from spowtd.functions._specific_yield import SpecificYield
from spowtd.functions.peat_growth import create_peat_growth_function

from spowtd.test.memory import repeat_create_destroy
from spowtd.test.utils import assert_close

# These values are not completely arbitrary --- we use the different regions of these
# curves to come up with simple problems that we can solve analytically.  This means
# making the transmissivity affine, and working with the low region where it is
# constant, and the high region where it is linear in zeta; making the specific yield
# constant; and working with the peat growth function either in the high region where
# productivity is uniform, or finding the steady state numerically and then verifying
# the ODE solver holds it steady.
production = 1.0
peat_growth = create_peat_growth_function(
    production=production, zeta_knots=[70.0, -400.0], k_knots=[1.0]
)
zeta_knots = np.linspace(-400.0, 400.0, 3)
specific_yield_scalar = 0.42
Sy_knots = np.array([specific_yield_scalar] * 3)
uniform_specific_yield = SpecificYield(zeta_knots, Sy_knots)

# Trick here: we use the region between knots, where conductivity is constant, or the
# region below the lowest knot, where transmissivity is constant
zeta_min_Tt = 50.0
Tt_zeta_knots = np.array([zeta_min_Tt, 300.0])
conductivity = 1.1
K_knots = np.array([conductivity] * 2)
min_Tt = 300.0
affine_transmissivity = TransmissivityNearSurface(
    zeta_knots=Tt_zeta_knots, K_knots=K_knots, minimum_transmissivity=min_Tt
)


@pytest.mark.parametrize(
    'solver_class, state_class',
    [
        (sim_mod.FixedStepStorageSurfaceSimulator0D, state_mod.FixedStepStorageState0D),
        (
            sim_mod.FixedStepWaterTableSurfaceSimulator0D,
            state_mod.FixedStepWaterTableState0D,
        ),
    ],
)
def test_none_arguments(solver_class, state_class):
    """None as an argument shall raise TypeError"""
    time = np.arange(4.0)
    trajectory = np.empty((4, 2), dtype='float64')
    trajectory[:] = np.nan
    trajectory[0, :] = (0, 0)
    recharge_scalar = 0.0
    recharge = ConstantScalarFunction(recharge_scalar)
    laplacian = -0.0023
    rhs = state_class(
        affine_transmissivity,
        peat_growth,
        uniform_specific_yield,
        recharge=recharge,
        laplacian=laplacian,
    )
    peat_growth_t = solver_class(rhs)
    kws = [{'time': time, 'trajectory': None}, {'time': None, 'trajectory': trajectory}]
    for kw in kws:
        with pytest.raises(TypeError):
            peat_growth_t.simulate(**kw)


@pytest.mark.parametrize(
    'solver_class, state_class, solver_kwargs',
    [
        (
            sim_mod.FixedStepStorageSurfaceSimulator0D,
            state_mod.FixedStepStorageState0D,
            {},
        ),
        (
            sim_mod.FixedStepWaterTableSurfaceSimulator0D,
            state_mod.FixedStepWaterTableState0D,
            {},
        ),
    ],
)
def test_simulate_linear(solver_class, state_class, solver_kwargs):
    """Test peat_growth_t simulation with constant transmissivity

    This is a simple test with constant transmissivity, constant
    specific yield, and constant recharge, so that zeta drops at a
    rate of (qn + Tt grad^2 p) / Sy

    """
    time = np.arange(5.0)
    trajectory = np.empty((5, 2), dtype='float64')
    trajectory[:] = np.nan
    zeta_0 = 1.3
    trajectory[0, :] = (zeta_0, 0.0)
    laplacian = -0.0023
    recharge_scalar = -0.2
    recharge = ConstantScalarFunction(recharge_scalar)
    rhs = state_class(
        affine_transmissivity,
        peat_growth,
        uniform_specific_yield,
        recharge=recharge,
        laplacian=laplacian,
    )
    peat_growth_t = solver_class(rhs, **solver_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        peat_growth_t.simulate(time, trajectory)
    zeta = trajectory[:, 0]
    expected_slope = (recharge_scalar + laplacian * min_Tt) / specific_yield_scalar
    expected_zeta = zeta_0 + np.arange(5) * expected_slope
    assert_close(expected_zeta, zeta)


@pytest.mark.parametrize(
    'solver_class, state_class, solver_kwargs',
    [
        (
            sim_mod.FixedStepStorageSurfaceSimulator0D,
            state_mod.FixedStepStorageState0D,
            {},
        ),
        (
            sim_mod.FixedStepWaterTableSurfaceSimulator0D,
            state_mod.FixedStepWaterTableState0D,
            {},
        ),
    ],
)
def test_simulate_constant_k(solver_class, state_class, solver_kwargs):
    """Test peat_growth_t simulation with constant conductivity

    This test has affine transmissivity
      Tt = Tmin + (z - z_min) K
    With constant specific yield and constant recharge, zeta has
    an exponential solution:
      z(t) = z(0) + exp(K L t / Sy)
                  - [1 - exp(K L t / Sy)] (Tmin / K + q / (KL) - zmin)
    where L = grad^2 p is the Laplacian.

    """
    time = np.arange(20.0) * 10
    trajectory = np.empty((20, 2), dtype='float64')
    trajectory[:] = np.nan
    zeta_0 = 102.0
    trajectory[0, :] = (zeta_0, 0.0)
    laplacian = -0.0023
    recharge_scalar = 1.2
    # we use the same transmissivity function, but above the lowest knot it is linear
    # instead of constant
    rhs = state_class(
        affine_transmissivity,
        peat_growth,
        uniform_specific_yield,
        recharge=ConstantScalarFunction(recharge_scalar),
        laplacian=laplacian,
    )
    peat_growth_t = solver_class(rhs, **solver_kwargs)
    peat_growth_t.simulate(time, trajectory)
    (
        zeta,  # pylint: disable=unpacking-non-sequence
        surface,
    ) = trajectory.T
    exp_term = np.exp(conductivity * laplacian * time / specific_yield_scalar)
    expected_zeta = zeta_0 * exp_term - (1 - exp_term) * (
        (min_Tt + recharge_scalar / laplacian) / conductivity - zeta_min_Tt
    )
    # rtol = 1e-4 for fixed-step solver
    assert_close(expected_zeta, zeta, rtol=1e-4)
    # For this test, the water table is high, so no decomposition
    if solver_class in (
        sim_mod.FixedStepStorageSurfaceSimulator0D,
        sim_mod.FixedStepWaterTableSurfaceSimulator0D,
    ):
        # Surface evolution not implemented within these classes
        expected_surface = [0.0] * len(time)
    else:
        expected_surface = time * production
    assert_close(expected_surface, surface)


@pytest.mark.parametrize(
    'solver_class, state_class',
    [
        (sim_mod.FixedStepStorageSurfaceSimulator0D, state_mod.FixedStepStorageState0D),
        (
            sim_mod.FixedStepWaterTableSurfaceSimulator0D,
            state_mod.FixedStepWaterTableState0D,
        ),
    ],
)
def test_sinusoidal_zeta(solver_class, state_class):
    """Sine test case generated with the Method of Manufactured Solutions

    Manufacture a sinusoidal oscillation of water level ζ
    ζ(t) = c sin(t/τ)
    by setting the forcing to
    q = Sy(ζ) c/τ cos(t/τ) + κ T(ζ)

    """
    # Configuration
    amp = 5.0
    tau = 10.0
    laplacian = -0.0023
    no_peat_growth = create_peat_growth_function(
        production=0.0, zeta_knots=[70.0, -400.0], k_knots=[0.0]
    )
    time = np.linspace(0, 2 * np.pi * tau, 20)
    solver_kwargs = {}

    def recharge_scalar(t):
        """Forcing that produces sinusoidal water level

        q = Sy(c sin(t/τ)) c/τ cos(t/τ) + κ T(c sin(t/τ))

        """
        return amp / tau * np.cos(t / tau) * uniform_specific_yield(
            amp * np.sin(t / tau)
        ) - laplacian * affine_transmissivity(amp * np.sin(t / tau))

    if solver_class in (
        sim_mod.FixedStepStorageSurfaceSimulator0D,
        sim_mod.FixedStepWaterTableSurfaceSimulator0D,
    ):
        # For these classes, integrate recharge for accuracy
        def recharge_scalar_integral(t0, t1):
            """Integral of recharge function"""
            result = scipy.integrate.quad(recharge_scalar, t0, t1)
            return result[0]
    else:
        recharge_scalar_integral = None

    recharge = PythonScalarFunction(recharge_scalar, recharge_scalar_integral)

    trajectory = np.empty((len(time), 2), dtype='float64')
    trajectory[:] = np.nan
    trajectory[0, :] = (0.0, 0.0)

    rhs = state_class(
        affine_transmissivity,
        no_peat_growth,
        uniform_specific_yield,
        recharge=recharge,
        laplacian=laplacian,
    )
    peat_growth_t = solver_class(rhs, **solver_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        peat_growth_t.simulate(time, trajectory)
    # pylint: disable=unpacking-non-sequence
    zeta, surface = trajectory.T
    assert_close(amp * np.sin(time / tau), zeta, rtol=1e-05, atol=1e-05)
    assert_close(np.zeros(time.shape), surface)


@pytest.mark.parametrize(
    'solver_class, state_class',
    [
        (sim_mod.FixedStepStorageSurfaceSimulator0D, state_mod.FixedStepStorageState0D),
        (
            sim_mod.FixedStepWaterTableSurfaceSimulator0D,
            state_mod.FixedStepWaterTableState0D,
        ),
    ],
)
def test_steady_state(solver_class, state_class):
    """Test peat_growth_t steady state with constant conductivity

    This test has affine transmissivity
      Tt = Tmin + (z - zmin) K
    With constant specific yield and constant recharge, so when
      dz / dt = 0,
      z = zmin - qn / (L K) - Tmin / K
    where L = grad^2 p is the Laplacian and qn is recharge.

    We calculate a recharge that makes the surface also a
    steady state, that is,
      qn = -L (Tmin + K (z* - zmin))
    where z* is the zeta giving zero peat growth (found by
    numerically inverting the peat growth function).

    """
    time = np.arange(5.0)
    trajectory = np.empty((5, 2), dtype='float64')
    trajectory[:] = np.nan
    laplacian = -0.0023
    zeta_0 = peat_growth.zero_growth_zeta()
    recharge_scalar = -laplacian * (min_Tt + (conductivity * (zeta_0 - zeta_min_Tt)))
    recharge = ConstantScalarFunction(recharge_scalar)
    trajectory[0, :] = (zeta_0, 0.0)
    # we use the same transmissivity function, but above the lowest knot it is linear
    # instead of constant
    rhs = state_class(
        affine_transmissivity,
        peat_growth,
        uniform_specific_yield,
        recharge=recharge,
        laplacian=laplacian,
    )
    peat_growth_t = solver_class(rhs)
    with warnings.catch_warnings(action='error'):
        peat_growth_t.simulate(time, trajectory)
    # pylint: disable=unpacking-non-sequence
    zeta, surface = trajectory.T
    expected_zeta = np.array([zeta_0] * 5)
    assert_close(zeta, expected_zeta)
    expected_surface = np.zeros(time.shape)
    assert_close(surface, expected_surface)


def test_create_destroy():
    """Making objects does not increase memory or ref counts"""
    transmissivity = affine_transmissivity
    specific_yield = uniform_specific_yield
    recharge = ConstantScalarFunction(0.0)
    rhs = state_mod.FixedStepStorageState0D(
        transmissivity,
        peat_growth,
        specific_yield,
        recharge=recharge,
        laplacian=-0.0023,
    )
    repeat_create_destroy(
        factory=sim_mod.FixedStepStorageSurfaceSimulator0D,
        refcounts={'stepper': 2},
        state=rhs,
    )
