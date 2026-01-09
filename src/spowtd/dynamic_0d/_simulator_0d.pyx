# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Simulators in 0D"""

from libc.math cimport NAN, INFINITY

import numpy as np
cimport numpy as np


cdef class FixedStepWaterTableSurfaceSimulator0D:
    """Simulator of peat dome hydrology

    end_state_weight is the weighting between values from the current and next time steps:
      * 0 means explicit (forward Euler)
      * 1 means fully implicit (backward Euler)
      * 0.5 (default) means Crank-Nicolson

    integrate_recharge is a boolean value determining whether the recharge for each step
    should be integrated (True, default) or if weighted values from the start and end of
    the step should be used (False).

    """
    state_class = FixedStepWaterTableState0D

    def __cinit__(self, FixedStepWaterTableState0D state not None,
                  double end_state_weight=0.5,
                  bint integrate_recharge=True,
                  nonlinear_solver='secant'):
        if nonlinear_solver == 'secant':
            self.stepper = FixedStepWaterTableStepper0D(
                state, end_state_weight, integrate_recharge)
        else:
            # XXX Check Cython docs; I think you are supposed to avoid throwing
            # XXX exceptions in __cinit__?
            raise ValueError(nonlinear_solver)

    def simulate(self,
                 np.ndarray[double, ndim=1, mode="c"] time not None,
                 np.ndarray[double, ndim=2, mode="c"] trajectory not None,
                 double end_state_weight=0.5):
        """Simulate water table and peat surface on a dome

        Trajectory is the output array into which the state trajectory is written; each
        row is a state comprising [zeta, surface].  On entry, the first row of the
        trajectory (trajectory[0, :]) must contain the initial conditions; on successful
        exit, the last row of trajectory (trajectory[-1, :]) will hold the final state.

        """
        if not trajectory.shape[1] == 2:
            raise ValueError('trajectory shape must be (n, 2), got ({}, {})'
                             .format(trajectory.shape[0],
                                     trajectory.shape[1]))
        if not time.shape[0] == trajectory.shape[0]:
            raise ValueError('time length {} != trajectory length {}'
                             .format(time.shape[0],
                                     trajectory.shape[0]))
        self.stepper.start_state.t = time[0]
        self.stepper.start_state.p = trajectory[0, 1]
        self.stepper.start_state.set_zeta(trajectory[0, 0])
        for i in range(1, len(time)):
            self.stepper.mixed_step(time[i])
            trajectory[i] = (self.stepper.end_state.zeta,
                             self.stepper.end_state.p)
            self.stepper.swap_states()


cdef class FixedStepStorageSurfaceSimulator0D:
    """Simulator of peat dome hydrology

    end_state_weight is the weighting between values from the current and next time steps:
      * 0 means explicit (forward Euler)
      * 1 means fully implicit (backward Euler)
      * 0.5 (default) means Crank-Nicolson

    integrate_recharge is a boolean value determining whether the recharge for each step
    should be integrated (True, default) or if weighted values from the start and end of
    the step should be used (False).

    """
    state_class = FixedStepStorageState0D

    def __cinit__(self, FixedStepStorageState0D state not None,
                  double end_state_weight=0.5,
                  bint integrate_recharge=True):
        self.stepper = FixedStepStorageStepper0D(
            state, end_state_weight, integrate_recharge)

    def simulate(self,
                 np.ndarray[double, ndim=1, mode="c"] time not None,
                 np.ndarray[double, ndim=2, mode="c"] trajectory not None,
                 double end_state_weight=0.5):
        """Simulate water table and peat surface on a dome

        Trajectory is the output array into which the state trajectory is written; each
        row is a state comprising [zeta, surface].  On entry, the first row of the
        trajectory (trajectory[0, :]) must contain the initial conditions; on successful
        exit, the last row of trajectory (trajectory[-1, :]) will hold the final state.

        """
        if not trajectory.shape[1] == 2:
            raise ValueError('trajectory shape must be (n, 2), got ({}, {})'
                             .format(trajectory.shape[0],
                                     trajectory.shape[1]))
        if not time.shape[0] == trajectory.shape[0]:
            raise ValueError('time length {} != trajectory length {}'
                             .format(time.shape[0],
                                     trajectory.shape[0]))
        self.stepper.start_state.t = time[0]
        self.stepper.start_state.p = trajectory[0, 1]
        self.stepper.start_state.set_storage(
            self.stepper.start_state
            .specific_yield
            .shallow_storage(trajectory[0, 0]))
        for i in range(1, len(time)):
            self.stepper.mixed_step(time[i])
            trajectory[i] = (self.stepper.end_state
                             .specific_yield
                             .zeta(self.stepper.end_state.Ws),
                             self.stepper.end_state.p)
            self.stepper.swap_states()
