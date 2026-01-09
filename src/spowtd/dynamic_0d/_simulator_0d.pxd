# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _simulator_0d extension module"""

from spowtd.dynamic_0d._state_0d cimport (
    FixedStepWaterTableState0D,
    FixedStepStorageState0D)
from spowtd.dynamic_0d._stepper_0d cimport (
    FixedStepWaterTableStepper0D,
    FixedStepStorageStepper0D)


cdef class FixedStepWaterTableSurfaceSimulator0D:
    cdef:
        readonly FixedStepWaterTableStepper0D stepper


cdef class FixedStepStorageSurfaceSimulator0D:
    cdef:
        readonly FixedStepStorageStepper0D stepper
