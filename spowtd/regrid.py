"""Regrid x, y points so that they are uniformly spaced on y

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def regrid(x, y, y_step, interpolant='linear'):
    """Interpolate points to a regular grid on the y axis

    Given a series of points x, y in which y is a function of x, yield a
    series of points (y, x) with y integer multiples of y_step representing a
    regridding of an interpolant of (x, y)

    In general there will be multiple points with the same y value and
    different xs if x |-> y is not one-to-one.

    interpolant may be any "kind" option acceptable to
    scipy.interpolant.interp1d, that is:
    linear, nearest, zero, slinear, quadratic, cubic

    Monotonicity of cubic splines is not ensured; this may cause problems.

    """
    # In between a pair of knots, find the xs corresponding to those
    #   discrete values of y by a root-finding algorithm.
    #
    # First, divide y by y_step so that integer values correspond to
    #   targets.  Then spline that.  Now, at each knot y_i, look for
    #   xs coorresponding to integer values of y on [y_i, y_i+1), or
    #   (y_i+1, y_i) depending on which is bigger.
    #
    if len(x) != len(y):
        raise ValueError(
            f'Argument lengths unequal: {len(x)} != {len(y)}'
        )
    # Don't use "if x" here, this is an ndarray
    if len(x) == 0:  # pylint: disable=len-as-condition
        return
    if not np.alltrue(np.isfinite(y)):
        raise ValueError('non-finite values in y vector')
    Y = y / y_step
    spline = interp1d(x, Y, kind=interpolant)
    y_int = np.array(np.ceil(Y), dtype=np.int64)
    for i in range(len(y_int) - 1):
        start, stop = (y_int[i], y_int[i + 1])
        if stop > start:
            targets = list(range(start, stop))
        else:
            targets = reversed(list(range(stop, start)))
        for y_target in targets:
            x_target = brentq(
                lambda x, y=y_target: spline(x) - y, x[i], x[i + 1]
            )
            yield (y_target, x_target)


if __name__ == '__main__':
    ys = np.array([2.0, 5.2, -1.3, -1.2, 10.0])
    xs = list(range(len(ys)))
    ys_step = 1.0
    new_points = list(regrid(xs, ys, ys_step, interpolant='nearest'))
    import matplotlib.pyplot as plt

    plt.plot(xs, ys, 'bo-')
    yp, xp = list(zip(*new_points))
    plt.plot(xp, yp, 'ro-')
    plt.show()
