# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Definitions for calling GSL's spline routines from Cython

See documentation here:
https://www.gnu.org/software/gsl/manual/html_node/Interpolation.html

"""

cimport numpy as np

from spowtd._scalar_function cimport ScalarFunction


cdef extern from "gsl/gsl_types.h":
    pass

cdef extern from "gsl/gsl_errno.h":
    ctypedef void gsl_error_handler_t(const char * reason, const char * file,
                                      int line, int gsl_errno)
    gsl_error_handler_t* gsl_set_error_handler_off()
    const char* gsl_strerror (const int gsl_errno)
    int GSL_EDOM

cdef extern from "gsl/gsl_interp.h":
    ctypedef struct gsl_interp_accel:
        pass
    gsl_interp_accel* gsl_interp_accel_alloc()
    void gsl_interp_accel_free(gsl_interp_accel * a)
    size_t gsl_interp_accel_find(gsl_interp_accel * a,
                                 const double x_array[],
                                 size_t size, double x)

    ctypedef struct gsl_interp_type:
        pass
    const gsl_interp_type* gsl_interp_linear
    const gsl_interp_type* gsl_interp_cspline

    ctypedef struct gsl_interp:
        pass

cdef extern from "gsl/gsl_spline.h":
    ctypedef struct gsl_spline:
          gsl_interp * interp
          double  * x
          double  * y
          size_t  size

    unsigned int gsl_interp_type_min_size(const gsl_interp_type * interp_type)
    gsl_interp * gsl_interp_alloc(const gsl_interp_type * T, size_t n)
    void gsl_interp_free(gsl_interp * interp)
    int gsl_interp_init(gsl_interp * obj,
                        const double xa[], const double ya[], size_t size)
    int gsl_interp_eval_e(const gsl_interp * obj,
                          const double xa[], const double ya[], double x,
                          gsl_interp_accel * a,
                          double * y)
    int gsl_interp_eval_deriv_e(const gsl_interp * obj,
                                const double xa[], const double ya[],
                                double x, gsl_interp_accel * a,
                                double * d)
    int gsl_interp_eval_deriv2_e(const gsl_interp * obj,
                                 const double xa[], const double ya[],
                                 double x, gsl_interp_accel * a,
                                 double * d2)
    int gsl_interp_eval_integ_e(const gsl_interp * obj,
                                const double xa[], const double ya[],
                                double a, double b,
                                gsl_interp_accel * acc,
                                double * result)


cdef class Interpolant(ScalarFunction):
    cdef:
        gsl_interp_accel* _accel
        gsl_interp* _interp
        size_t length
        readonly object x
        readonly object y
        double* xp
        double* yp

    cdef double call(self, double x) except -42
    cpdef double integrate(self, double a, double b) except -42
    cdef size_t find_bin(self, double x)
    cpdef update(self)
