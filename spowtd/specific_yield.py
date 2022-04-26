"""Specific yield classes

"""

import numpy as np

import scipy.stats

import spowtd.spline as spline_mod


def create_specific_yield_function(parameters):
    """Create a specific yield function

    Returns a callable object that returns specific yield at a given
    water level.  The class of the object depends on the "type" field
    in the parameters provided, and must be either "peatclsm" or
    "spline".

    """
    if 'type' not in parameters:
        raise ValueError(
            '"type" field is required in parameters; got {}'
            .format(parameters))
    sy_type = parameters.pop('type', None)
    return {
        'peatclsm': PeatclsmSpecificYield,
        'spline': SplineSpecificYield
    }[sy_type](**parameters)


class SplineSpecificYield:
    """Cubic spline representing specific yield

    zeta_knots_mm: Sequence of water levels in mm
    sy_knots: Specific yield values at those water levels

    """
    __slots__ = ['zeta_knots_mm', 'sy_knots', '_spline']

    def __init__(self, zeta_knots_mm, sy_knots):
        self.zeta_knots_mm = zeta_knots_mm
        self.sy_knots = sy_knots
        self._spline = spline_mod.Spline.from_points(
            zip(zeta_knots_mm, sy_knots),
            order=3)

    def __call__(self, water_level_mm):
        result = self._spline(water_level_mm)
        result[
            water_level_mm < self.zeta_knots_mm[0]] = self.sy_knots[0]
        result[
            water_level_mm > self.zeta_knots_mm[-1]] = self.sy_knots[-1]
        return result


class PeatclsmSpecificYield:
    """Specific yield function used in PEATCLSM

    sd:  standard deviation of microtopographic distribution, m
    theta_s:  saturated moisture content, mˆ3/ mˆ3
    b:  shape parameter, dimensionless
    psi_s:  air entry pressure, m

    """
    __slots__ = ['sd', 'theta_s', 'b', 'psi_s', '_spline',
                 'zeta_knots_mm', 'sy_knots']

    def __init__(self, sd, theta_s, b, psi_s):
        self.sd = sd
        self.theta_s = theta_s
        self.b = b
        self.psi_s = psi_s
        self.zeta_knots_mm = None
        self.sy_knots = None
        self._spline = self._construct_spline()

    def _construct_spline(self):
        """Construct specific yield spline

        """
        # Calculate the specific yield (Dettmann and Bechtold 2015,
        # Hydrological Processes)
        zl_ = np.linspace(-1, 1, 201)
        zu_ = np.linspace(-0.99, 1.01, 201)
        Sy1_soil = np.empty((201,), dtype='float64')
        Sy1_soil[:] = np.NaN
        self.get_Sy_soil(Sy1_soil, zl_, zu_)
        zeta_knots_m = 0.5 * (zu_ + zl_)
        Sy1_surface = scipy.stats.norm.cdf(
            zeta_knots_m, loc=0, scale=self.sd)
        self.sy_knots = Sy1_soil + Sy1_surface
        self.zeta_knots_mm = zeta_knots_m * 1000
        spline = spline_mod.Spline.from_points(
            zip(self.zeta_knots_mm, self.sy_knots),
            order=1)
        assert np.allclose(spline(self.zeta_knots_mm), self.sy_knots)
        return spline

    def __call__(self, water_level_mm):
        result = self._spline(water_level_mm)
        result[
            water_level_mm < self.zeta_knots_mm[0]] = self.sy_knots[0]
        result[
            water_level_mm > self.zeta_knots_mm[-1]] = self.sy_knots[-1]
        return result

    def get_Sy_soil(self, Sy_soil, zl_, zu_):
        """Calculate soil specific yield profile

        See equation 1 in Dettmann & Bechtold 2015, Hydrological Processes

        """
        theta_s = self.theta_s
        b = self.b
        psi_s = self.psi_s
        sd = self.sd
        zm_a = 0.5 * (zl_ + zu_)
        Fs_a = scipy.stats.norm.cdf(zm_a, loc=0, scale=sd)
        dz = zu_ - zl_
        for i in range(len(zl_)):
            zl = zl_[i]
            zu = zu_[i]
            A = 0
            for j in range(len(Sy_soil)):
                zm = zm_a[j]
                # Apply Campbell function to get soil moisture profile
                # for lower (zl) water level
                Azl = campbell_1d_az(Fs_a[j], zm, zl, theta_s, psi_s, b, sd)
                # Apply campbell function to get soil moisture profile
                # for upper (zu) water level
                Azu = campbell_1d_az(Fs_a[j], zm, zu, theta_s, psi_s, b, sd)
                A = A + dz[j] * (Azu - Azl)
            Sy_soil[i] = 1 / (1 * dz[i]) * A


def campbell_1d_az(Fs, z_, zlu, theta_s, psi_s, b, sd):
    """Soil moisture profile from Campbell function and microtopography

    See equations 4 and 5 in Dettmann & Bechtold 2015, Hydrological
    Processes

    """
    # PEATCLSM microtopographic distribution
    if ((zlu - z_) * 100) >= (psi_s * 100):
        theta = theta_s
    else:
        theta = theta_s * (((zlu - z_) * 100) / (psi_s * 100)) ** (-1 / b)
    theta_Fs = (1 - Fs) * theta
    return theta_Fs
