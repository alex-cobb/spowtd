# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Tests for spline"""

import bisect
import gc

import pytest

from numpy import allclose, arange, linspace, nan, sin

from scipy.integrate import quad

import spowtd._scalar_function as func_mod
from spowtd._spline import Interpolant, spline_to_tolerance
from spowtd.exceptions import OutOfDomainError
from spowtd.test.memory import repeat_create_destroy


x = arange(5, dtype='float64')
y = sin(x)


class TestLinearSpline:
    """Tests for spline class"""

    degree = 1

    def setup_class(self):
        """Fixture: Create linear spline"""
        # pylint: disable=attribute-defined-outside-init
        self.spline = Interpolant(x, y, self.degree)

    def test_is_scalar_function_subclass(self):
        """Instances are ScalarFunctions"""
        assert isinstance(self.spline, func_mod.ScalarFunction)

    def test_call(self):
        """Verify that values match at nodes"""
        yp = [self.spline(xv) for xv in x]
        assert allclose(y, yp), '{} not close to {}'.format(y, yp)

    def test_integrate(self):
        """Verify integrate method gives same result as quadrature"""
        integrals = [self.spline.integrate(x[i], x[i + 1]) for i in range(len(x) - 1)]
        quad_integrals = [
            quad(self.spline, x[i], x[i + 1])[0] for i in range(len(x) - 1)
        ]
        assert allclose(integrals, quad_integrals), '{} not close to {}'.format(
            integrals, quad_integrals
        )

    def test_find_bin(self):
        """Test lookup of interpolation interval for values"""
        for xv in (0, 0.1, 0.9, 1, 1.1, 1.5, 2, 2.1):
            i_ref = bisect.bisect(x, xv) - 1
            i = self.spline.index(xv)
            assert i == i_ref, 'index {} != {} for value {}'.format(i, i_ref, xv)

    @staticmethod
    def test_bad_degree():
        """Only degrees 1 and 3 are supported"""
        with pytest.raises(ValueError):
            Interpolant(x, y, 5)

    def test_out_of_domain(self):
        """Argument outside interpolation interval raises OutOfDomainError"""
        with pytest.raises(OutOfDomainError):
            for i in range(100):
                _ = (self.spline(x.min() - 1), self.spline(x.max() + 1))
                del i

    def test_x_out_of_order(self):
        """Constructor throws ValueError if x is not strictly increasing"""
        xp = x.copy()
        xp[0] = xp[2]
        with pytest.raises(ValueError):
            spline = Interpolant(xp, y, self.degree)
            del spline

    def test_nan_x(self):
        """Constructor throws ValueError if x has a nan"""
        xp = x.copy()
        xp[2] = nan
        with pytest.raises(ValueError):
            spline = Interpolant(xp, y, self.degree)
            del spline

    def test_nan_y(self):
        """Constructor throws ValueError if y has a nan"""
        yp = y.copy()
        yp[2] = nan
        with pytest.raises(ValueError):
            spline = Interpolant(x, yp, self.degree)
            del spline

    def test_different_sizes(self):
        """Constructor throws ValueError if x and y are not the same size"""
        xp = arange(len(x) + 1, dtype='float64')
        with pytest.raises(ValueError):
            spline = Interpolant(xp, y, self.degree)
            del spline

    def test_find_bin_out_of_domain(self):
        """find_bin for an out-of-domain value gives lowest or highest bin

        For a spline initialized with n values, the lowest bin is 0
        and the highest bin is n - 2

        """
        min_index = 0
        max_index = len(x) - 2
        assert self.spline.index(x.min()) == min_index
        assert self.spline.index(x.min() - 1) == min_index
        assert self.spline.index(x.max()) == max_index
        assert self.spline.index(x.max() + 1) == max_index

    def test_local_copy(self):
        """You can delete the data arrays used to construct the spline"""
        xp = arange(9, dtype='float64')
        yp = sin(xp)
        spline = Interpolant(xp, yp, self.degree)
        x_grid = linspace(xp.min(), xp.max(), 7)
        del xp
        del yp
        gc.collect()
        result = [spline(value) for value in x_grid]
        del result

    def test_create_destroy(self):
        """Making lots of splines does not increase memory or ref counts"""
        repeat_create_destroy(
            factory=Interpolant,
            refcounts={'x': 2, 'y': 2},
            x=x,
            y=y,
            degree=self.degree,
        )

    @staticmethod
    def test_spline_to_tolerance():
        """Test constructor for spline with specified tolerance"""
        xa = min(x)
        xb = max(x)
        max_rmse = 1e-8
        interpolant = spline_to_tolerance(
            sin, xa, xb, max_knots=2000, max_rmse=max_rmse
        )

        def squared_error(xv):
            """Evaluate squared difference between function and interpolant"""
            return (sin(xv) - interpolant(xv)) ** 2  # pylint: disable=not-callable

        rmse = quad(squared_error, xa, xb)[0]
        assert rmse <= max_rmse


class TestCubicSpline(TestLinearSpline):
    """Similar tests for cubic spline"""

    degree = 3
