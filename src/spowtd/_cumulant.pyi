# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Typing stubs for _cumulant"""

import numpy as np

class FullPrecisionCumulant:
    def reset(self, value: float = 0): ...
    def evaluate(self) -> float: ...

class Operator:
    def sum(self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float: ...
    def mean(self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float: ...
