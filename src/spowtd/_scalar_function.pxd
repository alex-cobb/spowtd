# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Scalar functions for simulations"""

cdef class ScalarFunction:
    cdef double call(self, double time) except? -42

    cpdef double integral(
        self, double start_time, double end_time) except? -42


cdef class PythonScalarFunction(ScalarFunction):
    cdef:
        object python_function
        object python_integral


cdef class ConstantScalarFunction(ScalarFunction):
    cdef:
        double scalar


cdef class PeriodicBoxcarFunction(ScalarFunction):
   cdef:
       double box_height
       double box_width
       double period
       double phase
       double baseline

   cdef size_t find_bin(self, double time) except? -1


cdef class PiecewiseConstantFunction(ScalarFunction):
    cdef:
        readonly object time
        readonly object value
        readonly double delta_t

    cdef size_t find_bin(self, double time) except -1
