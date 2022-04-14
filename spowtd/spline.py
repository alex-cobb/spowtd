"""Non-parametric 1D spline

"""

import scipy.interpolate as interpolate_mod
import numpy as np

splev = interpolate_mod.splev
splrep = interpolate_mod.splrep


class Spline:
    """Non-parametric cubic spline

    Splines are implemented using FITPACK, as wrapped in Scipy.
    FITPACK's tck parameters for the coordinate are stored and used
    for interpolation.

    """
    def __init__(self, tck):
        self._tck = tck

    @classmethod
    def from_points(cls, points, s=0, order=3):
        """Create a spline from points

        * points are (x, y) pairs.
        * s is the FITPACK smoothing parameter; 0 is for interpolation
          (default)
        * order is the order of the spline fit; 1 for linear
          interpolation, 3 for a cubic spline

        """
        x, y = zip(*points)
        if not np.isfinite(x).all():
            raise ValueError('non-finite value in x')
        if not np.isfinite(y).all():
            raise ValueError('non-finite value in y')
        if not (np.diff(x) > 0).all():
            raise ValueError('x must be strictly increasing')
        tck = splrep(x, y, s=s, k=order)
        return cls(tck)

    def domain(self):
        """End points of spline domain (x_start, x_end)

        """
        return (self._tck[0][0], self._tck[0][-1])

    def __call__(self, x, der=0):
        """Evaluate der'th derivative of spline at x

        """
        return splev(x, self._tck, der=der)
