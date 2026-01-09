# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test peat_growth extension module"""

import numpy as np

import spowtd.functions.peat_growth as peat_growth_mod
import spowtd.test.utils as test_utils_mod


assert_close = test_utils_mod.assert_close


class TestPeatGrowth:
    """Tests for peat growth calculation class"""

    alpha = 10.0**0.5
    production = 2.1
    zeta_min = -20.0
    zeta_max = 10.0

    def setup_class(self):
        """Fixture: Create peat growth function"""

        # pylint: disable=attribute-defined-outside-init
        self.function = peat_growth_mod.create_peat_growth_function(
            production=self.production,
            zeta_knots=[self.zeta_max, self.zeta_min],
            k_knots=[self.alpha],
        )

    def test_water_table_above_surface(self):
        """Verify that with zeta > zeta_max, peat growth = production"""

        for zeta_exp in np.linspace(-10, 10, 10):
            assert_close(self.function(self.zeta_max + 10**zeta_exp), self.production)

    def test_linear_region(self):
        """Test peat growth function where it is proportional to depth"""

        dpdt_min = self.function(self.zeta_min)
        for zeta in np.linspace(self.zeta_min, self.zeta_max, 10):
            dpdt = dpdt_min + (zeta - self.zeta_min) * self.alpha
            assert_close(self.function(zeta), dpdt)

    def test_dry_effect(self):
        """Verify that with zeta <= zeta_min, peat_growth(zeta) = peat_growth(zeta_min)"""

        dpdt_min = self.function(self.zeta_min)
        for zeta_exp in np.linspace(-10, 10, 10):
            zeta = self.zeta_min - 10**zeta_exp
            assert_close(self.function(zeta), dpdt_min)
