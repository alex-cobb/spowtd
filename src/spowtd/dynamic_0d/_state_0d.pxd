# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Interface to _state_0d extension module"""

from spowtd.functions._peat_growth cimport PeatGrowth
from spowtd.functions._specific_yield cimport SpecificYield
from spowtd.functions._transmissivity cimport Transmissivity
from spowtd._scalar_function cimport ScalarFunction


cdef class FixedStepWaterTableState0D:
    cdef:
        readonly Transmissivity transmissivity
        readonly SpecificYield specific_yield
        readonly PeatGrowth peat_growth
        public ScalarFunction recharge
        public double laplacian
        object solver_trajectory_file
        public double t
        public double zeta
        public double p
        public double T
        public double Sy

    cpdef copy(self)

    cpdef set_zeta(self, double zeta)


cdef class FixedStepStorageState0D:
    cdef:
        readonly Transmissivity transmissivity
        readonly SpecificYield specific_yield
        readonly PeatGrowth peat_growth
        public ScalarFunction recharge
        public double laplacian
        object solver_trajectory_file
        public double t
        readonly double Ws
        public double p
        public double T
        public double Sy

    cpdef copy(self)

    cpdef set_storage(self, double Ws)
