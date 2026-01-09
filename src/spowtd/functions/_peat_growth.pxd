# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _peat_growth extension module"""

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport numpy as np

from spowtd._spline cimport Interpolant


cdef class PeatGrowth:
    cdef:
        Interpolant _peat_production_spline
        double max_growth
        double min_growth
        double zeta_max
        double zeta_min
        double zeta_phi_1
        double alpha
    cdef double call(self, double zeta)
