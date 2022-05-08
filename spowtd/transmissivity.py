"""Transmissivity classes

"""

import numpy as np

import scipy.integrate as integrate_mod

import spowtd.spline as spline_mod


def create_transmissivity_function(parameters):
    """Create a transmissivity function

    Returns a callable object that returns transmissivity at a given
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
        'peatclsm': PeatclsmTransmissivity,
        'spline': SplineTransmissivity
    }[sy_type](**parameters)


class SplineTransmissivity:
    """Transmissivity parameterized as a spline of log conductivity

    zeta_knots_mm: Sequence of water levels in mm
    K_knots: Condutivity values at those water levels

    Stores a set of knots representing hydraulic conductivity at water
    table heights (relative to surface) zeta.  When called, takes a
    water table height and returns a transmissivity obtained by linear
    interpolation of log-conductivity.

    This is an extended value function that returns
    minimum_transmissivity below min(zeta) and extrapolates
    exponentially or linearly above max(zeta), according to whether
    the last two knots have the same or different conductivity.

    """
    __slots__ = ['zeta_knots_mm', 'K_knots_km_d',
                 'minimum_transmissivity_m2_d', '_spline']

    def __init__(self, zeta_knots_mm, K_knots_km_d,
                 minimum_transmissivity_m2_d):
        self.zeta_knots_mm = np.asarray(zeta_knots_mm, dtype='float64')
        self.K_knots_km_d = np.asarray(K_knots_km_d, dtype='float64')
        self.minimum_transmissivity_m2_d = minimum_transmissivity_m2_d
        log_K_knots = np.log(K_knots_km_d)
        self._spline = spline_mod.Spline.from_points(
            zip(zeta_knots_mm, log_K_knots),
            order=1)

    def conductivity(self, water_level_mm):
        assert water_level_mm >= self.zeta_knots_mm.min()
        if water_level_mm >= self.zeta_knots_mm.max():
            raise NotImplementedError('Extrapolation above highest knot')
        return np.exp(self._spline(water_level_mm))

    def __call__(self, water_level_mm):
        if np.isscalar(water_level_mm):
            return self.call_scalar(water_level_mm)
        return np.array(
            [self.call_scalar(value) for value in water_level_mm],
            dtype='float64')

    def call_scalar(self, water_level_mm):
        """Compute transmissivity for a scalar argument

        """
        if water_level_mm <= self.zeta_knots_mm.min():
            return self.minimum_transmissivity_m2_d
        return (
            self.minimum_transmissivity_m2_d +
            integrate_mod.quad(
                self.conductivity,
                self.zeta_knots_mm.min(),
                water_level_mm)[0])


class PeatclsmTransmissivity:
    """Transmissivity function used in PEATCLSM

    Computes transmissivity in m^2 / s from water level in mm.

    See equation 3 in Apers et al. 2022, JAMES.

    """
    __slots__ = ['Ksmacz0', 'alpha', 'zeta_max_cm']

    def __init__(self, Ksmacz0, alpha, zeta_max_cm):
        self.Ksmacz0 = Ksmacz0
        self.alpha = alpha
        self.zeta_max_cm = zeta_max_cm

    def __call__(self, water_level_mm):
        Ksmacz0 = self.Ksmacz0
        alpha = self.alpha
        zeta_max_cm = self.zeta_max_cm
        water_level_mm = np.asarray(water_level_mm)
        if (water_level_mm / 10 > zeta_max_cm).any():
            raise ValueError('T undefined at water level > {} cm in {}'
                             .format(zeta_max_cm, water_level_mm / 10))
        return (
            Ksmacz0 * (zeta_max_cm - water_level_mm / 10) ** (1 - alpha)
        ) / (100 * (alpha - 1))
