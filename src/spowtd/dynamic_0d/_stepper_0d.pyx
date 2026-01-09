# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Time stepper objects in 0D"""

from libc.math cimport isfinite, NAN

import numpy as np
cimport numpy as np

import scipy.optimize as optimize_mod

from spowtd.dynamic_0d._state_0d cimport FixedStepWaterTableState0D, FixedStepStorageState0D


cdef class FixedStepWaterTableStepper0D:
    """Control iteration through time to solve dynamic 0D problem

    end_state_weight is the weighting between values from the current and next time steps:
      * 0 means explicit (forward Euler)
      * 1 means fully implicit (backward Euler)
      * 0.5 (default) means Crank-Nicolson

    integrate_recharge is a boolean value determining whether the recharge for each step
    should be integrated (True, default) or if weighted values from the start and end of
    the step should be used (False).

    """
    def __cinit__(self, FixedStepWaterTableState0D state,
                  double end_state_weight=0.5,
                  bint integrate_recharge=True):
        self.start_state = state
        self.end_state = state.copy()
        # This class only advances the water table; p is unchanged.
        self.end_state.p = state.p
        self.end_state_weight = end_state_weight
        self.integrate_recharge = integrate_recharge
        self._rhs = NAN
        self._forward_recharge_depth = NAN
        self._backward_recharge_depth = NAN

    def __call__(self, double end_time):
        """Take a step to specified time

        This method exists for testing from Python; for use from
        Cython, see mixed_step()

        """
        self.mixed_step(end_time)

    cdef int mixed_step(self, double end_time) except -1:
        """Take a mixed step forward in time

        A mixed step uses a weighted average of values from the beginning and the end of
        the time step to compute the change in water table during the time step.

        end_state_weight is the weight to give to the end state in mixed step; for
          example, the effective diffusivity will be
          (1 - end_state_weight) * D(W^n) + end_state_weight * D(W^n+1)

        Returns the number of iterations (number of evaluations of end state) --- at
        least 1 on success, or -1 on error.

        """
        cdef:
            int iterations = -1
            double delta_t = end_time - self.start_state.t
        if delta_t <= 0:
            raise ValueError('Non-positive time step {}'
                             .format(delta_t))
        self.end_state.t = end_time
        # This class only updates the water level; p is untouched.
        self.end_state.p = self.start_state.p
        if self.integrate_recharge:
            self._forward_recharge_depth = (
                self.start_state.recharge.integral(
                    self.start_state.t, end_time))
            self._backward_recharge_depth = self._forward_recharge_depth
        else:
            self._forward_recharge_depth = (
                delta_t * self.start_state.recharge(self.start_state.t))
            self._backward_recharge_depth = (
                delta_t * self.start_state.recharge(end_time))
        self._update_rhs()
        # If this is a forward Euler step, the new zeta *is* the right-hand-side; we're
        #    done.
        if self.end_state_weight == 0:
            self.end_state.set_zeta(self._rhs)
            return 1
        # An implicit step
        # Initialize with unchanged zeta and p.
        self.end_state.zeta = self.start_state.zeta
        # Transfer T and Sy, instead of calling set_zeta(), to avoid recalculation
        self.end_state.T = self.start_state.T
        self.end_state.Sy = self.start_state.Sy
        root, r = optimize_mod.newton(self.compute_residual,
                                      x0=self.start_state.zeta,
                                      # XXX Use Ferziger and Peric conditions
                                      # rtol=1e-9,
                                      # xtol=1e-15,
                                      maxiter=1000,
                                      full_output=True)
        self.end_state.set_zeta(root)
        return(r.function_calls)

    cdef _update_rhs(self):
       """Update right-hand-side, which is fixed during iteration"""
       cdef:
           double zeta = self.start_state.zeta
           double kappa = -self.start_state.laplacian
           double T = self.start_state.T
           double Sy = self.start_state.Sy
           double f = self.end_state_weight
           double delta_t = self.end_state.t - self.start_state.t
           double delta_P = self._forward_recharge_depth
       assert kappa >= 0
       self._rhs = (zeta
                    - (1 - f) * delta_t * kappa * T / Sy
                    + (1 - f) * delta_P / Sy)

    def compute_residual(self, zeta):
        """Compute residual for scalar root-finding"""
        cdef:
           FixedStepWaterTableState0D end_state = self.end_state
           double kappa = -self.end_state.laplacian
           double f = self.end_state_weight
           double delta_t = self.end_state.t - self.start_state.t
           double recharge_depth = self._backward_recharge_depth

        if not isfinite(zeta):
            raise ValueError('Zeta not finite at t = {} ({})'
                             .format(self.end_state.t, zeta))
        end_state.set_zeta(zeta)
        return (self._rhs
                - f * delta_t * kappa * end_state.T / end_state.Sy
                + f * recharge_depth / end_state.Sy
                - zeta)

    def swap_states(self):
        """Swap start and end states"""
        cdef:
            FixedStepWaterTableState0D swap = None
        swap = self.start_state
        self.start_state = self.end_state
        self.end_state = swap


cdef class FixedStepStorageStepper0D:
    """Control iteration through time to solve dynamic 0D problem

    end_state_weight is the weighting between values from the current and next time steps:
      * 0 means explicit (forward Euler)
      * 1 means fully implicit (backward Euler)
      * 0.5 (default) means Crank-Nicolson

    integrate_recharge is a boolean value determining whether the recharge for each step
    should be integrated (True, default) or if weighted values from the start and end of
    the step should be used (False).

    """
    def __cinit__(self, FixedStepStorageState0D state,
                  double end_state_weight=0.5,
                  bint integrate_recharge=True):
        self.start_state = state
        self.end_state = state.copy()
        # This class only advances the water table; p is unchanged.
        self.end_state.p = state.p
        self.end_state_weight = end_state_weight
        self.integrate_recharge = integrate_recharge
        self._rhs = NAN
        self._forward_recharge_depth = NAN
        self._backward_recharge_depth = NAN

    def __call__(self, double end_time):
        """Take a step to specified time

        This method exists for testing from Python; for use from Cython, see
        mixed_step()

        """
        self.mixed_step(end_time)

    cdef int mixed_step(self, double end_time) except -1:
        """Take a mixed step forward in time

        A mixed step uses a weighted average of values from the beginning and the end of
        the time step to compute the change in water table during the time step.

        end_state_weight is the weight to give to the end state in mixed step; for
          example, the effective diffusivity will be
          (1 - end_state_weight) * D(W^n) + end_state_weight * D(W^n+1)

        Returns the number of iterations (number of evaluations of end state) --- at
        least 1 on success, or -1 on error.

        """
        cdef:
            int iterations = -1
            double delta_t = end_time - self.start_state.t
        if delta_t <= 0:
            raise ValueError('Non-positive time step {}'
                             .format(delta_t))
        self.end_state.t = end_time
        # This class only updates the water level; p is untouched.
        self.end_state.p = self.start_state.p
        if self.integrate_recharge:
            self._forward_recharge_depth = (
                self.start_state.recharge.integral(
                    self.start_state.t, end_time))
            self._backward_recharge_depth = self._forward_recharge_depth
        else:
            self._forward_recharge_depth = (
                delta_t * self.start_state.recharge(self.start_state.t))
            self._backward_recharge_depth = (
                delta_t * self.start_state.recharge(end_time))
        self._update_rhs()
        # If this is a forward Euler step, the new Ws *is* the right-hand-side; we're
        #    done.
        if self.end_state_weight == 0:
            self.end_state.set_storage(self._rhs)
            return 1
        # An implicit step.
        # Initialize with unchanged Ws and p
        self.end_state.Ws = self.start_state.Ws
        # Transfer T and Sy, instead of calling set_storage(), to avoid recalculation
        self.end_state.T = self.start_state.T
        self.end_state.Sy = self.start_state.Sy
        root, r = optimize_mod.newton(self.compute_residual,
                                      x0=self.start_state.Ws,
                                      # XXX Use Ferziger and Peric conditions
                                      # rtol=1e-9,
                                      # xtol=1e-15,
                                      maxiter=1000,
                                      full_output=True)
        self.end_state.set_storage(root)
        return(r.function_calls)

    cdef _update_rhs(self):
       """Update right-hand-side, which is fixed during iteration"""
       cdef:
           double Ws = self.start_state.Ws
           double kappa = -self.start_state.laplacian
           double T = self.start_state.T
           double Sy = self.start_state.Sy
           double f = self.end_state_weight
           double delta_t = self.end_state.t - self.start_state.t
           double delta_P = self._forward_recharge_depth
       assert kappa >= 0
       self._rhs = (Ws
                    - (1 - f) * delta_t * kappa * T
                    + (1 - f) * delta_P)

    def compute_residual(self, Ws):
        """Method to compute residual for KINSOL"""
        cdef:
           FixedStepStorageState0D end_state = self.end_state
           double kappa = -self.end_state.laplacian
           double f = self.end_state_weight
           double delta_t = self.end_state.t - self.start_state.t
           double recharge_depth = self._backward_recharge_depth

        if not isfinite(Ws):
            raise ValueError('State not finite at t = {} ({})'
                             .format(self.end_state.t, Ws))
        end_state.set_storage(Ws)
        return (self._rhs
                - f * delta_t * kappa * end_state.T
                + f * recharge_depth
                - Ws)

    def swap_states(self):
        """Swap start and end states"""
        cdef:
            FixedStepStorageState0D swap = None
        swap = self.start_state
        self.start_state = self.end_state
        self.end_state = swap
