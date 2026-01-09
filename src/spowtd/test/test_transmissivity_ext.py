# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test transmissivity extension module"""

import numpy as np

from scipy.integrate import quad

import spowtd.functions.transmissivity as transmissivity_mod
from spowtd.test.memory import repeat_create_destroy
from spowtd.test.utils import assert_close


class TestTransmissivity:
    """Test code for transmissivity calculation class"""

    zeta_knots = [-350, 6.49441445, 12.0, 180.0]
    # Include two identical values for uniform conductivity on one sub-interval
    K_knots = [5.34088667e-01, 7.19975731e-01, 7.19975731e-01, 6.77311672e03]

    def setup_class(self):
        """Fixture with roughly exponential transmissivity function"""
        # pylint: disable=attribute-defined-outside-init
        self.function = transmissivity_mod.create_transmissivity_function(
            type='spline', zeta_knots_mm=self.zeta_knots, K_knots_km_d=self.K_knots
        )

    def test_create_destroy(self):
        """Making transmissivity does not increase memory or ref counts"""
        repeat_create_destroy(
            factory=transmissivity_mod.create_transmissivity_function,
            refcounts={},
            type='spline',
            zeta_knots_mm=self.zeta_knots,
            K_knots_km_d=self.K_knots,
        )

    @staticmethod
    def test_uniform_conductivity():
        """Test special case of uniform conductivity"""
        K = 1.0
        function = transmissivity_mod.create_transmissivity_function(
            type='spline',
            zeta_knots_mm=np.array([10.0, 190.0]),
            K_knots_km_d=np.array([K, K]),
        )
        assert_close(function.conductivity(10.0), K, 'uniform K at lowest knot')
        assert_close(function.conductivity(100.0), K, 'uniform K between knots')
        assert_close(
            function.conductivity(210.0), K, 'uniform K extrapolated above top knot'
        )

    def test_knot_conductivities(self):
        """Verify conductivities at knots"""
        for i, zeta in enumerate(self.zeta_knots):
            assert_close(self.function.conductivity(zeta), self.K_knots[i])

    def test_transmissivities_in_domain(self):
        """Transmissivity matches quadrature of conductivity

        This test covers values within the interval between the lowest
        and highest knot.

        """
        zetas = np.linspace(self.zeta_knots[0], self.zeta_knots[-1], 100)
        Tt = [self.function(zeta) for zeta in zetas]
        quad_Tt = [
            quad(self.function.conductivity, self.zeta_knots[0], zeta)[0]
            for zeta in zetas
        ]
        assert_close(Tt, quad_Tt)

    def test_transmissivities_below_domain(self):
        """Calls below the lowest knot shall give a minimum transmissivity"""
        zetas = np.linspace(self.zeta_knots[0] - 200.0, self.zeta_knots[0], 100)
        Tt = [self.function(zeta) for zeta in zetas]
        Tt_min = self.function(self.zeta_knots[0])
        assert_close(Tt, Tt_min)

    def test_transmissivities_above_domain(self):
        """Transmissivity above the knots shall be extrapolated exponentially"""
        zetas = np.linspace(self.zeta_knots[-1], self.zeta_knots[-1] + 200, 100)
        Tt = [self.function(zeta) for zeta in zetas]
        quad_Tt = [
            quad(self.function.conductivity, self.zeta_knots[0], zeta)[0]
            for zeta in zetas
        ]
        assert_close(Tt, quad_Tt)


def test_constant_conductivity():
    """Test constant transmissivity and constant conductivity

    With two identical conductivities, one gets uniform
    transmissivity below the lowest knot and uniform conductivity
    above it.

    """
    Tt_min = 2**0.5
    threshold = 10.0
    conductivity = 1.25
    zeta_knots = np.array([threshold, 200.0])
    K_knots = np.array([conductivity, conductivity])
    Tt_func = transmissivity_mod.create_transmissivity_function(
        type='spline',
        zeta_knots_mm=zeta_knots,
        K_knots_km_d=K_knots,
        minimum_transmissivity_m2_d=Tt_min,
    )
    zetas = np.linspace(-400.0, 400.0, 100)
    Tt = [Tt_func(zeta) for zeta in zetas]
    quad_Tt = [
        Tt_min + quad(Tt_func.conductivity, zeta_knots[0], zeta)[0] for zeta in zetas
    ]
    assert_close(quad_Tt, Tt)
    below_threshold = np.linspace(threshold - 400.0, threshold, 100)
    Tt = [Tt_func(zeta) for zeta in below_threshold]
    ref = Tt_min
    assert_close(ref, Tt)
    above_threshold = np.linspace(threshold, threshold + 400.0, 100)
    Tt = [Tt_func(zeta) for zeta in above_threshold]
    ref = [Tt_min + (zeta - threshold) * conductivity for zeta in above_threshold]
    assert_close(ref, Tt)
