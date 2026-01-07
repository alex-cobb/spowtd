# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Typing stubs for _spline"""

from typing import Callable

from spowtd._scalar_function import ScalarFunction

class Interpolant(ScalarFunction):
    def __call__(self, x: float) -> float: ...
    def integrate(self, a: float, b: float) -> float: ...
    def index(self, x: float) -> int: ...

def spline_to_tolerance(
    function: Callable,
    xa: float,
    xb: float,
    max_rmse: float = 1e-8,
    max_knots: int = 1000,
) -> Interpolant: ...
