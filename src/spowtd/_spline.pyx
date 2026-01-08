# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Class to manage the lifetime of a spline

"""

from libc.math cimport INFINITY

import numpy as np
cimport numpy as np

from scipy.integrate import quad

from spowtd.exceptions import GSLError, OutOfDomainError


cdef class Interpolant(ScalarFunction):
    """Interpolant between points using GSL

    A local copy of the data in the input vectors is held by the
    interpolant.

    """
    def __cinit__(self,
                  np.ndarray[double, ndim=1, mode="c"] x not None,
                  np.ndarray[double, ndim=1, mode="c"] y not None,
                  unsigned int degree):
        cdef:
            size_t length = x.shape[0]
            unsigned int min_size = 0
            const gsl_interp_type* interp_type_p = NULL
            np.ndarray[double, ndim=1, mode="c"] own_x
            np.ndarray[double, ndim=1, mode="c"] own_y

        # Disable the default GSL error handler (core dump) and never
        # restore; as recommended in the library, we check return codes.
        # https://www.gnu.org/software/gsl/manual/html_node/\
        #   Error-Handling.html
        gsl_set_error_handler_off()

        if <size_t>y.shape[0] != length:
            raise ValueError('x and y vectors must have the same length')
        if not np.isfinite(x).all():
            raise ValueError('non-finite value in x')
        if not np.isfinite(y).all():
            raise ValueError('non-finite value in y')
        if not (np.diff(x) > 0).all():
            raise ValueError('x must be strictly increasing')
        self.length = length

        if degree == 1:
            interp_type_p = gsl_interp_linear
        elif degree == 3:
            interp_type_p = gsl_interp_cspline
        else:
            raise ValueError('degree must be 1 (linear) or 3 (cubic), '
                             'got {}'.format(degree))
        min_size = gsl_interp_type_min_size(interp_type_p)
        if length < min_size:
            raise ValueError('sequence length {} < minimum {} '
                             'for interpolant of degree '
                             '{}'.format(length, min_size, degree))
        own_x = np.empty(length, 'float64')
        own_y = np.empty(length, 'float64')
        own_x[:] = x
        own_y[:] = y
        self.x = own_x
        self.y = own_y
        self.xp = &own_x[0]
        self.yp = &own_y[0]
        self._interp = gsl_interp_alloc(interp_type_p, length)
        if self._interp is NULL:
            raise MemoryError()
        self._accel = gsl_interp_accel_alloc()
        self.update()

    cpdef update(self):
        """Update interpolant to any changes in x and y arrays

        The update does not verify that x and y are still finite and
        that x is strictly ascending.  If those assumptions are
        violated, behaviour is undefined.

        """
        gsl_interp_init(self._interp, self.xp, self.yp, self.length)

    def __dealloc__(self):
        if self._interp is not NULL:
            gsl_interp_free(self._interp)
        if self._accel is not NULL:
            gsl_interp_accel_free(self._accel)

    cdef double call(self, double x) except -42:
        """Get interpolated value of y at specified x

        """
        cdef:
            int errno
            double y
            const double* xp = self.xp
            const double* yp = self.yp
        errno = gsl_interp_eval_e(self._interp, xp, yp, x, self._accel, &y)
        if errno == 0:
            return y
        elif errno == GSL_EDOM:
            raise OutOfDomainError(x)
        else:
            raise GSLError(gsl_strerror(errno))

    cpdef double integrate(self, double a, double b) except -42:
        """Compute integral of y from a through b

        """
        cdef:
            int errno
            double integral
            double x_min, x_max
            const double* xp = self.xp
            const double* yp = self.yp

        errno = gsl_interp_eval_integ_e(self._interp, xp, yp, 
                                        a, b, self._accel,
                                        &integral)
        if errno == 0:
            return integral
        elif errno == GSL_EDOM:
            x_min = xp[0]
            x_max = xp[self.length - 1]
            raise OutOfDomainError("integration interval [{}, {}] outside "
                                   "domain [{}, {}]".format(a, b,
                                                            x_min, x_max))
        else:
            raise GSLError(gsl_strerror(errno))

    def __call__(self, double x):
        """Get interpolated value of y at specified x

        For calls from C, use o.call() instead.  This method exists
        primarily for testing from Python.

        """
        return self.call(x)

    cdef size_t find_bin(self, double x):
        """Find the index of the bin for x

        Returns an index i such that x_array[i] <= x < x_array[i+1]

        """
        cdef:
            const double* xp = self.xp
        return gsl_interp_accel_find(self._accel, xp, self.length, x)

    def index(self, double x):
        """Python wrapper for find_bin, for testing

        """
        return self.find_bin(x)


def spline_to_tolerance(function, double xa, double xb,
                        double max_rmse=1e-8, size_t max_knots=1000):
    """Create a spline of function to specified rmse tolerance

    Splines the function with a sufficient number of knots that root
    mean squared error is no greater than max_rmse.  Starting with
    four points, the spline is created, and the root-mean-squared
    error computed on the interval xa, xb by quadrature; if the rmse
    constraint is not satisfied, the number of points is doubled and
    the procedure is repeated.

    """
    cdef:
        size_t num_knots = 2
        double rmse = INFINITY

    def squared_error(x):
        """Compute square of difference between interpolant and function

        """
        return (function(x) - interpolant(x)) ** 2

    while not rmse <= max_rmse:
        num_knots *= 2
        if num_knots > max_knots:
            raise ValueError("rmse {} > {} with {} knots"
                             "".format(rmse, max_rmse, num_knots))
        knots = np.linspace(xa, xb, num_knots)
        interpolant = Interpolant(knots,
                                  np.array([function(knot)
                                            for knot in knots]),
                                  degree=3)
        # full_output=1 suppresses warnings about reaching the maximum
        # number of subdivisions of an interval
        rmse = (quad(squared_error, xa, xb,
                     full_output=1)[0] / (xb - xa)) ** 0.5
    return interpolant
