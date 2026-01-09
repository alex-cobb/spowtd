# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Functions for peat properties"""

cimport numpy as np
import numpy as np
from scipy.optimize import brentq


cdef class PeatGrowth:
    """Parameterized peat growth function

    Parameters are specified in __cinit__

    """
    def __cinit__(self, Interpolant peat_production_spline, double alpha, double zeta_min):
        cdef:
            double* zeta = peat_production_spline.xp
            double* growth = peat_production_spline.yp
            size_t size = peat_production_spline.length
        self.alpha = alpha
        self.zeta_min = zeta_min
        self._peat_production_spline = peat_production_spline
        self.zeta_phi_1 = zeta[0]
        self.min_growth = growth[0]
        self.zeta_max = zeta[size - 1]
        self.max_growth = growth[size - 1]
        if not self.zeta_min <= self.zeta_phi_1:
            raise ValueError('zeta_phi_1 {} < zeta_min {}'
                             .format(self.zeta_phi_1, self.zeta_min))
        if not self.zeta_phi_1 < self.zeta_max:
            raise ValueError('zeta_max {} <= zeta_phi_1 {}'
                             .format(self.zeta_max, self.zeta_phi_1))
        if not self.max_growth >= self.min_growth:
            raise ValueError('max growth {} < min_growth {}'
                             .format(self.max_growth, self.min_growth))

    def __call__(self, double zeta):
        return self.call(zeta)

    cdef double call(self, double zeta):
        """Compute the peat accumulation or loss rate"""

        if zeta >= self.zeta_max:
            return self.max_growth
        elif zeta >= self.zeta_phi_1:
            return self._peat_production_spline(zeta)
        elif zeta >= self.zeta_min:
            return self.min_growth - self.alpha * (self.zeta_phi_1 - zeta)
        else:  # zeta < self.zeta_min
            return self.min_growth - self.alpha * (self.zeta_phi_1
                                                   - self.zeta_min)

    def zero_growth_zeta(self):
        """Find water table elevation giving zero growth

        Uses Brent's algorithm.
        Equivalent to scipy.optimize.brentq(peat_growth,
                                            peat_growth.zeta_min,
                                            peat_growth.zeta_max)

        """
        return brentq(self, self.zeta_min, self.zeta_max)
