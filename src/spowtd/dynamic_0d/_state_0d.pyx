# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Surface and hydrologic state in 0d simulation"""

from libc.math cimport isfinite, NAN, INFINITY

import sys


cdef class FixedStepWaterTableState0D:
    """Water level and surface

    Maintains dependent quantities.

    """
    def __cinit__(self, Transmissivity transmissivity,
                  PeatGrowth peat_growth,
                  SpecificYield specific_yield,
                  ScalarFunction recharge,
                  double laplacian,
                  solver_trajectory_file=None):
        self.transmissivity = transmissivity
        self.peat_growth = peat_growth
        self.specific_yield = specific_yield
        self.recharge = recharge
        self.laplacian = laplacian
        self.solver_trajectory_file = solver_trajectory_file
        self.t = NAN
        self.zeta = NAN
        self.p = NAN
        self.T = NAN
        self.Sy = NAN

    cpdef copy(self):
        """Create a shallow copy of the state object

        Transmissivity, peat growth, and specific yield functions are
        copied.

        t, zeta and p are not copied.

        """
        return self.__class__(
            self.transmissivity, self.peat_growth,
            self.specific_yield, self.recharge,
            self.laplacian, self.solver_trajectory_file)

    cpdef set_zeta(self, double zeta):
        """Set water level zeta and update dependent quantities

        Computes transmissivity and specific yield

        """
        cdef:
            double dzetadt = NAN
            double dpdt = NAN
        # Only reevaluate functions if zeta has changed
        if zeta == self.zeta:
            return
        self.zeta = zeta
        # Only makes sense for TransmissivityNearSurface,
        # so we pass NAN as peat thickness.
        self.T = self.transmissivity.call(zeta, NAN)
        self.Sy = self.specific_yield.call(zeta)
        # Could make this more efficient by subclassing; Maybe not
        # worth it, though, as probably won't use this class for
        # production code.
        if self.solver_trajectory_file is not None:
            dzetadt = ((self.recharge.call(self.t) + self.laplacian *
                        # Only makes sense for TransmissivityNearSurface,
                        # so we pass NAN as peat thickness.
                        self.transmissivity.call(zeta, NAN))
                       / self.specific_yield.call(zeta))
            dpdt = self.peat_growth.call(zeta)
            # Store data as strings; this allows infinity to be passed
            # through to Postgres.
            self.solver_trajectory_file.write(
                '["%s", "%s", "%s", "%s", "%s", "%s"]\n' %
                (self.t, zeta, self.p,
                 dzetadt, dpdt, self.recharge.call(self.t)))


cdef class FixedStepStorageState0D:
    """Water level and surface

    Maintains dependent quantities.

    """
    def __cinit__(self, Transmissivity transmissivity,
                  PeatGrowth peat_growth,
                  SpecificYield specific_yield,
                  ScalarFunction recharge,
                  double laplacian,
                  solver_trajectory_file=None):
        self.transmissivity = transmissivity
        self.peat_growth = peat_growth
        self.specific_yield = specific_yield
        self.recharge = recharge
        self.laplacian = laplacian
        self.solver_trajectory_file = solver_trajectory_file
        self.t = NAN
        self.p = NAN
        self.T = NAN
        self.Sy = NAN
        self.Ws = NAN

    cpdef copy(self):
        """Create a shallow copy of the state object

        Transmissivity, peat growth, and specific yield functions are
        copied.

        t, Ws and p are not copied.

        """
        return self.__class__(
            self.transmissivity, self.peat_growth,
            self.specific_yield, self.recharge,
            self.laplacian, self.solver_trajectory_file)

    cpdef set_storage(self, double Ws):
        """Set storage and update dependent quantities

        """
        cdef:
            double zeta = NAN
            double dzetadt = NAN
            double dpdt = NAN
        # Only reevaluate functions if storage has changed
        if Ws == self.Ws:
            return
        self.Ws = Ws
        zeta = self.specific_yield.zeta(Ws)
        # Only makes sense for TransmissivityNearSurface,
        # so we pass NAN as peat thickness.
        self.T = self.transmissivity.call(zeta, NAN)
        self.Sy = self.specific_yield.call(zeta)
        # Could make this more efficient by subclassing; Maybe not
        # worth it, though, as probably won't use this class for
        # production code.
        if self.solver_trajectory_file is not None:
            dzetadt = ((self.recharge.call(self.t) + self.laplacian *
                        # Only makes sense for TransmissivityNearSurface,
                        # so we pass NAN as peat thickness.
                        self.transmissivity.call(zeta, NAN))
                       / self.specific_yield.call(zeta))
            dpdt = self.peat_growth.call(zeta)
            # Store data as strings; this allows infinity to be passed
            # through to Postgres.
            self.solver_trajectory_file.write(
                '["%s", "%s", "%s", "%s", "%s", "%s"]\n' %
                (self.t, zeta, self.p,
                 dzetadt, dpdt, self.recharge.call(self.t)))
