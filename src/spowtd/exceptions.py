# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Package-specific exceptions"""


class ConvergenceError(Exception):
    """Indicates that an iterative procedure has failed to converge"""


class SingularityError(Exception):
    """Apparent singularity in function"""


class OutOfDomainError(Exception):
    """Error thrown when argument is outside function domain"""


class GSLError(Exception):
    """Error raised by GNU Scientific Library"""
