# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _specific_yield extension module"""

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np

from spowtd._spline cimport Interpolant


cdef class SpecificYield:
    cdef:
        public Interpolant _spline
        Interpolant _zeta_spline
        double _zeta_min
        double _zeta_max
        double _Sy_min
        double _Sy_max
        double _shallow_storage_min
        double _shallow_storage_max
    cdef double call(self, double zeta) except -1
    cpdef double shallow_storage(self, double zeta) except? -42
    cpdef double deep_storage(self, double surface) except? -42
    cpdef double zeta(self, double shallow_storage) except? -42
    cpdef double storage(self, double water_table,
                         double surface) except? -42
