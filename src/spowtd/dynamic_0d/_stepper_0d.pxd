# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _stepper_0d extension module"""

from spowtd.dynamic_0d._state_0d cimport FixedStepWaterTableState0D, FixedStepStorageState0D


cdef class FixedStepWaterTableStepper0D:
    cdef:
        double end_state_weight
        FixedStepWaterTableState0D start_state
        FixedStepWaterTableState0D end_state
        bint integrate_recharge
        double _rhs
        double _forward_recharge_depth
        double _backward_recharge_depth

    cdef int mixed_step(self, double end_time) except -1
    cdef _update_rhs(self)


cdef class FixedStepStorageStepper0D:
    cdef:
        double end_state_weight
        FixedStepStorageState0D start_state
        FixedStepStorageState0D end_state
        bint integrate_recharge
        double _rhs
        double _forward_recharge_depth
        double _backward_recharge_depth

    cdef int mixed_step(self, double end_time) except -1
    cdef _update_rhs(self)
