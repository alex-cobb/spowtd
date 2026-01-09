# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Typing stubs for _simulator_0d"""

from ._stepper_0d import FixedStepWaterTableStepper0D, FixedStepStorageStepper0D

class FixedStepWaterTableSurfaceSimulator0D:
    stepper: FixedStepWaterTableStepper0D

class FixedStepStorageSurfaceSimulator0D:
    stepper: FixedStepStorageStepper0D
