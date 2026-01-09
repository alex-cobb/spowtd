# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Elapsed time and calendar handling for simulations"""


class ElapsedDaysIterator:
    """Yields elapsed time in days on a regular grid

    Instances of this class are generators, but have a length that can be queried with
    len().

    Times are given in days since midnight on January 1st of the start year in the
    proleptic Gregorian calendar.

    """

    def __init__(self, time_step_days, n_steps):
        self.time_step_d = time_step_days
        self.n_steps = n_steps
        self.i = -1

    def __len__(self):
        return self.n_steps - self.i - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i < self.n_steps:
            return self.i * self.time_step_d
        raise StopIteration
