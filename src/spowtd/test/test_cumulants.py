# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for full-precision cumulant

Tests local implementation against the lsum implementation from the ASPN receipe from
which it was derived:
http://code.activestate.com/recipes/393090/

"""

from math import frexp
import random

from spowtd._cumulant import FullPrecisionCumulant


# Always use the same seed for repeatability.  As discussed in the comments on the
# recipe, there are rare cases where lsum may be inexact because of the use of the
# platform's strtod.  Of course, it will still be far, far better than a naive sum.
SEED = 42


class TestFullPrecisionCumulant:
    """Tests for exact cumulant"""

    @staticmethod
    def test_against_hettinger():
        """Test full precision cumulant using test in Hettinger's recipe"""
        random.seed(SEED)
        for j in range(250):
            del j
            vals = [7, 1e100, -7, -1e100, -9e-20, 8e-20] * 10
            s = 0
            for i in range(200):
                del i
                v = random.gauss(0, random.random()) ** 7 - s
                s += v
                vals.append(v)
            random.shuffle(vals)
            cumulant = FullPrecisionCumulant()
            for v in vals:
                cumulant += v
            assert lsum(vals) == cumulant.evaluate(), '{} != {}'.format(
                lsum(vals), cumulant.evaluate()
            )

    @staticmethod
    def test_tim_peters_example():
        """Test full precision cumulant using example from Tim Peters

        Given in comments to Hettinger's recipe

        """
        expected_sum = 20000.0
        cumulant = FullPrecisionCumulant()
        for v in [1, 1e100, 1, -1e100] * 10000:
            cumulant += v
        assert cumulant.evaluate() == expected_sum


def lsum(iterable):
    """Full precision summation using long integers for intermediate values

    This is the implementation from APSPN recipe 393090

    """
    # Transform (exactly) a float to m * 2 ** e where m and e are integers.
    # Adjust (tmant,texp) and (mant,exp) to make texp the common exponent.
    # Given a common exponent, the mantissas can be summed directly.

    tmant, texp = 0, 0
    for x in iterable:
        mant, exp = frexp(x)
        mant, exp = int(mant * 2.0**53), exp - 53
        if texp > exp:
            tmant <<= texp - exp
            texp = exp
        else:
            mant <<= exp - texp
        tmant += mant
    return float(str(tmant)) * 2.0**texp
