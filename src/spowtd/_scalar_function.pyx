# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Scalar function objects

Uniform interface for different scalar functions used for recharge and
boundary conditions in dynamic simulations.  The independent variable
is called "time" but could be anything.

"""

from libc.math cimport NAN, ceil, floor, INFINITY, isfinite

cimport numpy as np
import numpy as np


cdef class ScalarFunction:
    cdef double call(self, double time) except? -42:
        raise NotImplementedError

    cpdef double integral(self, double start_time,
                          double end_time) except? -42:
        raise NotImplementedError

    def next_discontinuity(self, double time):
        return (INFINITY, NAN)

    def __call__(self, double time):
        """Evaluate scalar function at specified time

        """
        return self.call(time)


cdef class PythonScalarFunction(ScalarFunction):
    def __cinit__(self,
                  object python_function,
                  object python_integral):
        self.python_function = python_function
        self.python_integral = python_integral

    cdef double call(self, double time) except? -42:
        return self.python_function(time)

    cpdef double integral(self, double start_time,
                          double end_time) except? -42:
        return self.python_integral(start_time, end_time)


cdef class ConstantScalarFunction(ScalarFunction):
    def __cinit__(self, double scalar):
        self.scalar = scalar

    cdef double call(self, double time) except? -42:
        return self.scalar

    cpdef double integral(self, double start_time,
                          double end_time) except? -42:
        return self.scalar * (end_time - start_time)


cdef class PeriodicBoxcarFunction(ScalarFunction):
    def __cinit__(self,
                  double box_height, double box_width,
                  double period, double phase,
                  double baseline=0.0):
        self.box_height = box_height
        self.box_width = box_width
        self.period = period
        self.phase = phase
        self.baseline = baseline

    cdef double call(self, double time) except? -42:
        cdef:
            double period = self.period
            size_t i = self.find_bin(time)
            double period_start = NAN
            bint in_box
        period_start = i * period
        assert period_start <= time
        in_box = ((period_start + self.phase <= time) and
                  (time < period_start + self.box_width + self.phase))
        if in_box:
            return self.baseline + self.box_height
        return self.baseline

    def next_discontinuity(self, double time):
        cdef:
            double period = self.period
            double t_discontinuity = NAN
            double continuation = NAN
            size_t i = self.find_bin(time)
            double period_start =  NAN
        period_start = i * period
        assert period_start <= time
        if time < period_start + self.phase:
            t_discontinuity = period_start + self.phase
            continuation = self.baseline
        elif time < period_start + self.box_width + self.phase:
            t_discontinuity = period_start + self.phase + self.box_width
            continuation = self.baseline + self.box_height
        else:  # time - period_start >= self.box_width + self.phase
            t_discontinuity = (i + 1) * self.period + self.phase
            continuation = self.baseline
        assert t_discontinuity > time
        assert isfinite(t_discontinuity)
        assert isfinite(continuation)
        return (t_discontinuity, continuation)

    cpdef double integral(self, double start_time,
                          double end_time) except? -42:
        # Boxes start at i * period + phase, and end at
        #                i * period + phase + box_width
        # for all integers i.
        # After we subtract off phase from start_time and
        # end_time, boxes start at i * period and end at
        #                          i * period + box_width
        # for all integers i.
        cdef:
            double time = NAN
            double cumulative_width = 0
            double period = self.period
            double phase = self.phase
            double box_width = self.box_width
            size_t start_i = -1
            size_t end_i = -1
        if end_time <= start_time:
            raise ValueError('Not implemented: '
                             'end time {} <= start time {}'
                             .format(end_time, start_time))
        # Remove phase
        start_time -= phase
        end_time -= phase
        start_i = self.find_bin(start_time)
        end_i = self.find_bin(end_time)
        cumulative_width = (end_i + 1 - start_i) * box_width
        assert cumulative_width >= box_width, (
            end_i + 1, start_i)
        # Subtract off any portion of first box that is before start_time
        if start_time - (start_i * period) < box_width:
            cumulative_width -= (start_time - (start_i * period))
        else:
            cumulative_width -= box_width
        # Subtract off any portion of last box that is after end_time
        if end_time - (end_i * period) < box_width:
            cumulative_width -= box_width - (end_time - (end_i * period))
        assert cumulative_width >= 0, (
            'Cumulative width {} < 0; '
            'Period = {}, '
            'Start time = {}, '
            'End time = {}, '
            'Start time % period = {}, '
            'End time % period = {}, '
            'Box width = {}'.format(cumulative_width,
                                    period,
                                    start_time,
                                    end_time,
                                    start_time % period,
                                    end_time % period,
                                    box_width))
        return (self.baseline * (end_time - start_time) +
                cumulative_width * self.box_height)

    cdef size_t find_bin(self, double time) except? -1:
        """Find index i in time grid such that t[i] <= time < t[i + 1]

        If time lies before first grid time, Value Error is raised.
        If time lies after last grid time t[n - 1], n - 1 is returned.

        """
        cdef:
            double period = self.period
            size_t i = <size_t>(time // period)
        if not i * period <= time:
            i -= 1
        elif not (i + 1) * period > time:
            i += 1
        assert i * period <= time
        assert (i + 1) * period > time
        return i

    def __call__(self, double time):
        """Evaluate periodic boxcar function at specified time

                   Period
                   │────────>│
         ─ ─ ─     ┌─┐       ┌─┐
           ^       │ │       │ │
        Height   ──┘ └───────┘ └─────
                │─>│       ─>│ │<─
                Phase       Width

        """
        return self.call(time)


cdef class PiecewiseConstantFunction(ScalarFunction):
    """Piecewise constant function

    It is assumed that the time step is uniform.

    """
    def __cinit__(self,
                  np.ndarray[double, ndim=1, mode="c"] time not None,
                  np.ndarray[double, ndim=1, mode="c"] value not None):
        if time.shape[0] != value.shape[0]:
            raise ValueError('Lengths of arrays do not match: {} != {}'
                             .format(time.shape[0], value.shape[0]))
        if not (np.diff(time) > 0).all():
            raise ValueError('Time not strictly increasing in {}'
                             .format(time))
        self.delta_t = time[1] - time[0]
        if not np.allclose(self.delta_t, np.diff(time)):
            raise ValueError('Time step not uniform in {}'
                             .format(time))
        self.time = time
        self.value = value

    cdef double call(self, double time) except? -42:
        cdef:
            np.ndarray[double, ndim=1, mode="c"] v = self.value
        return v[self.find_bin(time)]

    cpdef double integral(self, double start_time,
                          double end_time) except? -42:
        cdef:
            np.ndarray[double, ndim=1, mode="c"] t = self.time
            np.ndarray[double, ndim=1, mode="c"] v = self.value
            size_t n = self.time.shape[0]
            size_t start_i = self.find_bin(start_time)
            size_t end_i = self.find_bin(end_time)
            size_t i = -1
            double delta_t = self.delta_t
            double integral = 0.0
        if end_time < start_time:
            raise NotImplementedError('End time {} precedes start time {}'
                                      .format(end_time, start_time))
        assert end_i >= start_i
        assert start_i < n
        assert end_i < n
        if start_i == end_i:
            return (end_time - start_time) * v[start_i]
        assert end_i > start_i
        assert start_i < n - 1
        assert t[start_i + 1] > start_time
        integral = (t[start_i + 1] - start_time) * v[start_i]
        for i in range(start_i + 1, end_i):
            integral += v[i] * delta_t
        assert end_time >= t[end_i]
        integral += (end_time - t[end_i]) * v[end_i]
        return integral

    cdef size_t find_bin(self, double time) except -1:
        """Find index i in time grid such that t[i] <= time < t[i + 1]

        If time lies before first grid time, Value Error is raised.
        If time lies after last grid time t[n - 1], n - 1 is returned.

        """
        cdef:
            np.ndarray[double, ndim=1, mode="c"] t = self.time
            size_t i = -1
            size_t n = self.time.shape[0]
            double delta_t = self.delta_t
        if time < t[0]:
            raise ValueError('Time {} precedes first point (t {})'
                             .format(time, t[0]))
        i = <size_t>((time - t[0]) // delta_t)
        if i >= n:
            i = n - 1
        assert i >= 0, i
        assert i < n, '{} < {}'.format(i, n)
        # Handle roundoff error
        if time < t[i]:
            i -= 1
        elif i != n - 1 and t[i + 1] <= time:
            i += 1
        # /Handle roudoff error
        assert t[i] <= time, '{} <= {}'.format(t[i], time)
        assert i == n - 1 or t[i + 1] > time, (
            '{} == {} or {} > {}'.format(i, n - 1, t[i + 1], time))
        return i

    def next_discontinuity(self, double time):
        """Get the next time of a discontinuity, and a continuation

        Returns the time of the next discontinuity, if any, and a
        value for smooth continuation of the function at the
        discontinuity.

        If no further discontinuities will occur, returns (inf, nan).

        """
        cdef:
            np.ndarray[double, ndim=1, mode="c"] t = self.time
            np.ndarray[double, ndim=1, mode="c"] v = self.value
            size_t n = self.time.shape[0]
            size_t i = self.find_bin(time)
        # XXX Might want to be able to set a threshold here.
        assert i <= n - 1, i
        if i == n - 1:
            return (INFINITY, NAN)
        return (t[i + 1], v[i])
