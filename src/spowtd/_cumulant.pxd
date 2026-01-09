# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _cumulant extension module"""

cdef class FullPrecisionCumulant:
    """Use long integers to maintain an exact cumulant

    Modifed from the lsum routine in ASPN recipe 393090 by Raymond Hettinger
    http://code.activestate.com/recipes/393090/

    """
    cdef:
        # A C long is not long enough for this; need Python's (arbitrarily long) ints
        object tmant
        long int texp


cdef class Operator:
    """Full-precision operations on vectors"""

    cdef:
        FullPrecisionCumulant cumulant
        size_t i
        size_t n
