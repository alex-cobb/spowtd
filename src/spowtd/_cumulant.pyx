# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Cumulants to store statistics about simulations"""

from libc.math cimport frexp

cimport numpy as np
import numpy as np


cdef class FullPrecisionCumulant:
    """Use long integers to maintain an exact cumulant

    Modifed from the lsum routine in ASPN recipe 393090 by Raymond Hettinger
    http://code.activestate.com/recipes/393090/

    """
    def __cinit__(self, double value=0):
        self.tmant = 0L
        self.texp = 0
        if value != 0:
            self += value

    def reset(self, double value=0):
        self.tmant = 0L
        self.texp = 0
        if value != 0:
            self += value

    def __iadd__(self, double x):
        """In-place add a floating point number at full precision"""

        cdef:
            int exp
            double mant
            object lmant
        mant = frexp(x, &exp)
        lmant = int(mant * 2.0 ** 53)
        exp -= 53
        if self.texp > exp:
            self.tmant <<= self.texp - exp
            self.texp = exp
        else:
            lmant <<= exp - self.texp
        self.tmant += lmant
        return self

    def evaluate(self):
        """Evaluate the cumulant"""

        return float(str(self.tmant)) * 2.0 ** self.texp


cdef class Operator:
    """Full-precision operations on vectors"""

    def __cinit__(self):
        self.cumulant = FullPrecisionCumulant.__new__(FullPrecisionCumulant, 0)

    def sum(self, np.ndarray[double, ndim=1, mode="c"] vector):
        """Compute full-precision sum of vector"""

        cdef:
            FullPrecisionCumulant cumulant = self.cumulant
            const double* vp = &vector[0]
            size_t i = 0
            size_t n = vector.shape[0]
        cumulant.tmant = 0L
        cumulant.texp = 0
        for i in range(n):
            cumulant.__iadd__(vp[i])
        return cumulant.evaluate()

    def mean(self, np.ndarray[double, ndim=1, mode="c"] vector):
        """Compute full-precision mean of vector"""

        return self.sum(vector) / <double>(vector.shape[0])
