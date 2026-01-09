# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Utility to create peat growth function"""

import numpy as np

# pylint: disable=no-name-in-module,unused-import
from spowtd._spline import Interpolant
from spowtd.functions._peat_growth import PeatGrowth


def create_peat_growth_function(production, zeta_knots, k_knots):
    """Create a peat growth function

    Given a scalar rate of peat production (at maximum water table), a sequence of water
    table heights ("zeta knots"), and a sequence of decomposition rate coefficients ("k
    knots"), create a callable object that interpolates a rate of peat accumulation or
    loss given a scalar water table height argument.  The decomposition rate
    coefficients represent decomposition rates on each interval *between* two water
    table height knots, and therefore there must be exactly one more zeta knot than the
    number of k knots.  If passed a water table height above the highest knot / below
    the lowest knot, the callable will return the rate of peat accumulation / loss at
    the highest knot (= productivity) / lowest knot, respectively.

    Knots are converted to Numpy double arrays as needed.

    """
    zeta = np.asarray(zeta_knots, dtype='float64')
    del zeta_knots
    k_knots = np.asarray(k_knots, dtype='float64')
    dpdt_knots = np.empty(zeta.shape, dtype='float64')
    dpdt_knots[:] = float(production)
    del production
    dpdt_knots[1:] += (k_knots * np.diff(zeta)).cumsum()
    sort_indices = np.argsort(zeta)
    zeta = zeta[sort_indices]
    assert (np.diff(zeta) >= 0).all()
    dpdt_knots = dpdt_knots[sort_indices]
    peat_production_spline = Interpolant(zeta, dpdt_knots, degree=1)
    alpha = k_knots[-1]
    return PeatGrowth(peat_production_spline, alpha, zeta.min())
