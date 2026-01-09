# -*- coding: utf-8 -*-

# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Transmissivity functions

Transmissivity is an abstract base class.  Two implementations are
provided:

1. TransmissivityNearSurface: Transmissivity is a function of water level ("zeta")
   relative to surface, obtained by linear interpolation of log-conductivity. Peat
   thickness does not matter.  Below a certain level, the transmissivity is constant
   (possibly 0).

2. TransmissivityAboveFloor: Transmissivity is the integral of conductivity above a
   floor.  Below the floor, conductivity is 0; above the floor, conductivity is a
   function of water level, obtained by linear interpolation of log-conductivity.

"""

from libc.math cimport exp, isfinite, log, INFINITY, NAN
cimport libc.math

cimport numpy as np
import numpy as np


cdef class Transmissivity:
    """Transmissivity function"""

    def __call__(self, double zeta, double thickness):
        return self.call(zeta, thickness)

    cdef double call(self, double zeta,
                     double thickness) except -1:
        """Calculate transmissivity (m^2 / d) from water level

        zeta is height above surface in mm
        thickness is height of surface above floor in mm.

        """
        raise NotImplementedError('Deferred to subclasses')

    cdef double conductivity_d(self, double zeta,
                               double thickness) except -1:
        """Compute conductivity (mm / d) from water level

        zeta is height above surface in mm
        thickness is height of surface above floor in mm.

        """
        raise NotImplementedError('Deferred to subclasses')

    def conductivity(self, double zeta, double thickness):
        return self.conductivity_d(zeta, thickness)


cdef class ConstantTransmissivity(Transmissivity):
    """Constant transmissivity

    Always returns the same value, regardless of arguments.

    """
    def __cinit__(self, transmissivity):
        self.transmissivity = transmissivity

    cdef double call(self, double zeta, double thickness) except -1:
        return self.transmissivity

    cdef double conductivity_d(self, double zeta,
                               double thickness) except -1:
        return 0.0


cdef class LogInterpolantTransmissivity(Transmissivity):
    """Parameterized transmissivity function

    Integral of a linear interpolant for log(K)

    """
    def __cinit__(self, np.ndarray[double, ndim=1, mode="c"] zeta_knots,
                  np.ndarray[double, ndim=1, mode="c"] K_knots,
                  double minimum_transmissivity=0.0):
        """Build a function for ∫ y dy using linear interpolation of log y(x)

        This is an extended value function that returns minimum_transmissivity below
        min(zeta) and extrapolates exponentially or linearly above max(zeta), according
        to whether the last two knots have the same or different conductivity.

        """
        self._size = zeta_knots.shape[0]
        if <size_t>K_knots.shape[0] != self._size:
            raise ValueError('len(zeta knots) {} != '
                             'len(K knots) {}'.format(self._size,
                                                      K_knots.shape[0]))
        self._zeta_min = zeta_knots[0]
        self._zeta_max = zeta_knots[self._size - 1]
        self._K_knots = <double*> PyMem_Malloc((self._size) *
                                               sizeof(double))
        if not self._K_knots:
            raise MemoryError()
        self._lambda = <double*> PyMem_Malloc((self._size) *
                                              sizeof(double))
        if not self._lambda:
            raise MemoryError()
        self._segment_integrals = <double*> PyMem_Malloc((self._size) *
                                                         sizeof(double))
        if not self._segment_integrals:
            raise MemoryError()

    def __dealloc__(self):
        PyMem_Free(self._K_knots)
        PyMem_Free(self._lambda)
        PyMem_Free(self._segment_integrals)

    def __init__(self, np.ndarray[double, ndim=1, mode="c"] zeta_knots,
                 np.ndarray[double, ndim=1, mode="c"] K_knots,
                  double minimum_transmissivity=0.0):
        """Function initialization

        Some more complicated operations happen here to cache some state for quick
        method calls.

        """
        cdef:
            size_t i = -1
            np.ndarray[double, ndim=1, mode="c"] log_K_knots

        log_K_knots = np.empty((self._size,))
        log_K_knots[:] = NAN
        for i in range(self._size):
            if K_knots[i] <= 0:
                raise ValueError('conductivity above lowest knot must be positive')
            self._K_knots[i] = K_knots[i]
            log_K_knots[i] = log(K_knots[i])
        self._spline = Interpolant.__new__(Interpolant,
                                           zeta_knots,
                                           log_K_knots,
                                           degree=1)
        self._cache_segment_integrals(minimum_transmissivity)

    def _cache_segment_integrals(self, double minimum_transmissivity):
        """Cache the (easy-to-compute) integrals up to knots

        Also caches length scales describing behaviour of y between knots.  This is kept
        as a side effect to ensure that segment integrals are still calculated correctly
        if y knots change.

        """
        cdef:
            size_t i = -1
            double integral = minimum_transmissivity
            double* lambdap = self._lambda
            double* integralp = self._segment_integrals
            double* zeta = self._spline.xp
            double* K_knots = self._K_knots

        self._cache_lambdas()
        integralp[0] = integral
        for i in range(1, self._size):
            if K_knots[i] != K_knots[i - 1]:
                integral += lambdap[i - 1] * (K_knots[i] - K_knots[i - 1])
            else:
                integral += K_knots[i] * (zeta[i] - zeta[i - 1])
            integralp[i] = integral

    cdef _cache_lambdas(self):
        """Cache length scales describing conductivity between knots

        """
        cdef:
            size_t i = -1
            double* zeta = self._spline.xp
            double* log_K = self._spline.yp

        # For convenience, we use lambda[i] for the length scale on
        # the interval [zeta_i, zeta_i+1].  lambda[size - 1] should
        # never be touched.
        for i in range(self._size - 1):
            if log_K[i + 1] != log_K[i]:
                self._lambda[i] = ((zeta[i + 1] - zeta[i]) /
                                   (log_K[i + 1] - log_K[i]))
            else:
                self._lambda[i] = INFINITY
        self._lambda[self._size - 1] = NAN

    cdef double _interpolate(self, double zeta) except -1:
        """Calculate near-surface transmissivity (m^2 / d) from water level

        zeta is height above surface in mm

        """
        cdef:
            size_t index = -1
            double K = NAN
            double K_at_knot_below = NAN
            double Tt_at_knot_below = NAN
            double partial_integral = NAN
            double* zeta_knots = self._spline.xp

        if zeta < self._zeta_min:
            return self._segment_integrals[0]
        elif zeta >= self._zeta_max:
            # out of range, extrapolate
            index = self._size - 2
        else:
            index = self._spline.find_bin(zeta)
            if not isfinite(zeta):
                raise ValueError('non-finite zeta {}'.format(zeta))
            assert zeta_knots[index] <= zeta < zeta_knots[index + 1], \
                '{} <= {} < {}'.format(zeta_knots[index], zeta,
                                       zeta_knots[index + 1])
        K_at_knot_below = self._K_knots[index]
        K = self._interpolate_conductivity_d(zeta)
        if K != K_at_knot_below:
            partial_integral = self._lambda[index] * (K - K_at_knot_below)
        else:
            partial_integral = K * (zeta - zeta_knots[index])
        Tt_at_knot_below = self._segment_integrals[index]
        return Tt_at_knot_below + partial_integral

    cdef double _interpolate_conductivity_d(self, double zeta) except -1:
        """Compute conductivity (mm / d) from water level

        zeta is height above surface in mm

        """
        cdef:
            double* log_K_knots = self._spline.yp

        assert zeta >= self._zeta_min
        if zeta > self._zeta_max:
            if isfinite(self._lambda[self._size - 2]):
                # Extrapolate exponentially above uppermost knot
                return exp(log_K_knots[self._size - 1]
                           + (zeta - self._zeta_max)
                           / self._lambda[self._size - 2])
            else:
                # Extrapolate linearly above uppermost knot (uniform conductivity)
                return self._K_knots[self._size - 1]
        return exp(self._spline.call(zeta))


cdef class TransmissivityNearSurface(LogInterpolantTransmissivity):
    """Transmissivity near surface

    """
    def __call__(self, double zeta, double thickness=NAN):
        return self.call(zeta, thickness)

    cdef double call(self, double zeta,
                     double thickness) except -1:
        """Calculate transmissivity (m^2 / d) from water level

        zeta is height above surface in mm
        thickness is height of surface above floor in mm.

        """
        return self._interpolate(zeta)

    def conductivity(self, double zeta, double thickness=NAN):
        """Compute conductivity (mm / d) from water level

        zeta is height above surface in mm
        thickness is height of surface above floor in mm.

        Thickness is ignored; it is included for compatibility of signature.

        """
        return self.conductivity_d(zeta, thickness)

    cdef double conductivity_d(self, double zeta,
                               double thickness) except -1:
        """Compute conductivity (mm / d) from water level

        zeta is height above surface in mm
        thickness is height of surface above floor in mm.

        Thickness is ignored; it is included for compatibility of signature.

        """
        if zeta < self._zeta_min:
            return 0.0
        return self._interpolate_conductivity_d(zeta)


cdef class TransmissivityAboveFloor(LogInterpolantTransmissivity):
    """Transmissivity above floor

    """
    cdef double call(self, double zeta,
                     double thickness) except -1:
        """Calculate transmissivity (m^2 / d) from water level

        zeta is height of water table above surface in mm;
        thickness is height of surface above floor in mm.

        Conductivity is treated as constant from the lowest knot down to the floor.

        """
        if thickness < 0:
            raise ValueError('Negative thickness {}'
                             .format(thickness))
        if -thickness >= zeta:
            # Water table is below floor.
            return 0
        if zeta <= self._zeta_min:
            # Water table is below lowest knot.  Floor is below water table, and
            # therefore floor is also below lowest knot.  Transmissivity is deep
            # conductivity times water table height above floor.
            return self._K_knots[0] * (zeta - -thickness)
        if -thickness < self._zeta_min:  # zeta > _zeta_min
            # Floor < lowest knot < water table.
            # Transmissivity is deep conductivity times height of lowest knot above
            # floor, plus integrated conductivity from lowest knot to water table.
            return (self._K_knots[0] * (self._zeta_min - -thickness) +
                    (self._interpolate(zeta) -
                     self._interpolate(self._zeta_min)))
        # Floor and water table are above lowest knot.
        return self._interpolate(zeta) - self._interpolate(-thickness)

    cdef double conductivity_d(self, double zeta,
                               double thickness) except -1:
        """Compute conductivity (mm / d) from water level

        zeta is height above surface in mm
        thickness is height of surface above floor in mm.

        Conductivity is treated as constant from the lowest knot down to the floor, and
        zero below the floor.

        """
        if zeta < -thickness:
            return 0.0
        if zeta < self._zeta_min:
            return self._K_knots[0]
        return self._interpolate_conductivity_d(zeta)
