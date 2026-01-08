# -*- coding: utf-8 -*-

# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Specific yield function"""

from libc.math cimport NAN
cimport libc.math

cimport numpy as np
import numpy as np
from scipy.optimize import brentq

from spowtd._spline import spline_to_tolerance


cdef class SpecificYield:
    """Parameterized specific yield function"""
    def __cinit__(self,
                  np.ndarray[double, ndim=1, mode="c"] zeta_knots,
                  np.ndarray[double, ndim=1, mode="c"] Sy_knots):
        self._zeta_min = zeta_knots[0]
        self._zeta_max = zeta_knots[zeta_knots.shape[0] - 1]
        self._Sy_min = Sy_knots[0]
        self._Sy_max = Sy_knots[Sy_knots.shape[0] - 1]
        self._spline = Interpolant.__new__(Interpolant,
                                           zeta_knots,
                                           Sy_knots,
                                           degree=3)
        self._shallow_storage_min = self.shallow_storage(self._zeta_min)
        self._shallow_storage_max = self.shallow_storage(self._zeta_max)
        self._zeta_spline = spline_to_tolerance(self._invert_for_zeta,
                                                self._shallow_storage_min,
                                                self._shallow_storage_max,
                                                max_knots=8192,
                                                max_rmse=1e-8)

    cpdef double storage(self, double water_table, double surface) except? -42:
        """Compute the integral of specific yield

        H is the water table elevation, p the peat surface elevation.  The integral is
        always computed from zero elevation, so in pathological cases it could be
        negative --- this is expected because the datum and the depth for storage are
        both arbitrary.

        """
        cdef:
            double zeta_1 = NAN
            double zeta_0 = NAN
            # deep_storage accounts for water below the bottom of the domain of specific
            # yield.  This is the storage in the deep peat, below fluctuations in the
            # water table observed by us.
            double storage = NAN
            bint downward_integral

        if water_table < 0:
            downward_integral = True
            zeta_0 = water_table - surface
            zeta_1 = -surface
        else:
            downward_integral = False
            zeta_0 = -surface
            zeta_1 = water_table - surface
        # These are the possibilities:
        # - ζ_1 < ζ_min, no intersection between [ζ_0, ζ_1] and [ζ_min, ζ_max]
        # - ζ_0 > ζ_max, no intersection
        # - ζ_0 < ζ_min:
        #   - ζ_1 < ζ_max
        #   - ζ_1 >= ζ_max
        # - ζ_0 >= ζ_min:
        #   - ζ_1 < ζ_max
        #   - ζ_1 >= ζ_max
        # We deal with the first two possiblities first, they are fast
        if zeta_1 <= self._zeta_min:
            storage = (zeta_1 - zeta_0) * self._Sy_min
        elif zeta_0 >= self._zeta_max:
            storage = (zeta_1 - zeta_0) * self._Sy_max
        elif zeta_0 < self._zeta_min:
            storage = (self._zeta_min - zeta_0) * self._Sy_min
            if zeta_1 < self._zeta_max:
                storage += self._spline.integrate(self._zeta_min,
                                                  zeta_1)
            else:  # zeta_1 >= self._zeta_max
                storage += self._spline.integrate(self._zeta_min,
                                                  self._zeta_max)
                storage += (zeta_1 - self._zeta_max) * self._Sy_max
        else:  # zeta_0 >= self._zeta_min
            if zeta_1 < self._zeta_max:
                storage = self._spline.integrate(zeta_0, zeta_1)
            else:  # zeta_1 >= self._zeta_max
                storage = self._spline.integrate(zeta_0, self._zeta_max)
                storage += (zeta_1 - self._zeta_max) * self._Sy_max
        if downward_integral:
            return -storage
        else:
            return storage

    cpdef double shallow_storage(self, double zeta) except? -42:
        """Compute the shallow storage

        Shallow storage is the integral of specific yield from zeta = 0 to zeta.

        """
        return self.storage(zeta, 0)

    cpdef double deep_storage(self, double surface) except? -42:
        """Compute the deep storage

        Deep storage is the integral of specific yield from zeta = -surface to 0.

        """
        return self.storage(surface, surface)

    cpdef double zeta(self, double shallow_storage) except? -42:
        """Compute the water table height zeta from shallow storage

        zeta is found via the inverse of the specific yield function, or equivalently,
        as the integral of the multiplicative inverse of specific yield.

        """
        if shallow_storage < self._shallow_storage_min:
            return self._zeta_min + ((shallow_storage
                                      - self._shallow_storage_min)
                                     / self._Sy_min)
        if shallow_storage > self._shallow_storage_max:
            return self._zeta_max + ((shallow_storage
                                      - self._shallow_storage_max)
                                     / self._Sy_max)
        return self._zeta_spline(shallow_storage)

    # This must be a Python method so that it can be passed to
    # spline_to_tolerance
    def _invert_for_zeta(self, double shallow_storage):
        """Find zeta corresponding to storage in the interpolated range

        Uses Brent's method to find zeta from shallow_storage(zeta) by one-dimensional
        root-finding.

        """
        return brentq(self._error, self._zeta_min, self._zeta_max,
                      args=(shallow_storage,))

    # This must be a Python method so that it can be passed to the
    # implementation of Brent's method in SciPy
    def _error(self, double zeta, double shallow_storage):
        """Compute the difference from the target storage at zeta

        This method exists for finding zeta from shallow_storage by one-dimensional root
        finding.

        """
        return self.shallow_storage(zeta) - shallow_storage

    def __call__(self, double zeta):
        return self.call(zeta)

    cdef double call(self, double zeta) except -1:
        """Calculate specific yield from water table height

        zeta is height above surface in mm

        """
        ## Spline will throw an exception if outside interpolation
        ## interval
        if zeta < self._zeta_min:
            return self._Sy_min
        elif zeta > self._zeta_max:
            return self._Sy_max
        return self._spline.call(zeta)
