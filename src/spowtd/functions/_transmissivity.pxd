# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _transmissivity extension module

"""

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np

from spowtd._spline cimport Interpolant


cdef class Transmissivity:
    cdef double call(self, double zeta, double thickness) except -1
    cdef double conductivity_d(self, double zeta,
                               double thickness) except -1


cdef class ConstantTransmissivity(Transmissivity):
    cdef:
        double transmissivity


cdef class LogInterpolantTransmissivity(Transmissivity):
    cdef:
        public Interpolant _spline
        size_t _size
        double _zeta_min
        double _zeta_max
        double* _lambda
        double* _segment_integrals
        double* _K_knots
    cdef double call(self, double zeta, double thickness) except -1
    cdef double _interpolate(self, double zeta) except -1
    cdef double conductivity_d(self, double zeta,
                               double thickness) except -1
    cdef double _interpolate_conductivity_d(self, double zeta) except -1
    cdef _cache_lambdas(self)


cdef class TransmissivityNearSurface(LogInterpolantTransmissivity):
    pass


cdef class TransmissivityAboveFloor(LogInterpolantTransmissivity):
    pass
