"""Test specific yield extension module"""

import numpy as np

from scipy.integrate import quad

from spowtd.functions._specific_yield import SpecificYield

import spowtd.test.memory as test_memory_mod
import spowtd.test.utils as test_utils_mod


get_memory_usage_kb = test_memory_mod.get_memory_usage_kb
assert_close = test_utils_mod.assert_close


class TestSpecificYield:
    """Test code for specific yield calculation class"""

    zeta_knots = [-180.0, -40, 0, 50, 120]
    Sy_knots = [0.1, 0.5, 0.3, 0.7, 0.2]

    def setup_class(self):
        """Fixture: create specific yield function"""
        # pylint: disable=attribute-defined-outside-init
        self.function = SpecificYield(
            np.asarray(self.zeta_knots, dtype='float64'),
            np.asarray(self.Sy_knots, dtype='float64'),
        )

    def test_knot_values(self):
        """Verify values at knots"""
        for i, zeta in enumerate(self.zeta_knots):
            assert_close(self.function(zeta), self.Sy_knots[i])

    def test_values_outside_domain(self):
        """Test values outside the interval spanned by the knots

        The lowermost or uppermost value should be returned when below
        or above the domain of the knots, respectively.

        """
        for zeta_exp in np.linspace(-10, 10, 10):
            low_zeta = self.zeta_knots[0] - 10**zeta_exp
            assert_close(self.function(low_zeta), self.Sy_knots[0])
            high_zeta = self.zeta_knots[-1] + 10**zeta_exp
            assert_close(self.function(high_zeta), self.Sy_knots[-1])

    def test_storage(self):
        """Verify that the storage() method corresponds to quadrature"""
        zetas = np.linspace(-300, 300, 100)
        surface = np.linspace(10.0, 200.0, 20)
        shallow_storage = [self.function.shallow_storage(zeta) for zeta in zetas]
        quad_shallow = [quad(self.function, 0, zeta)[0] for zeta in zetas]
        assert_close(
            quad_shallow,
            shallow_storage,
            'shallow_storage calculation matches quadrature',
        )
        for p in surface:
            deep_storage = self.function.deep_storage(p)
            quad_deep = quad(self.function, -p, 0)[0]
            assert_close(
                quad_deep, deep_storage, 'deep_storage calculation matches quadrature'
            )
            storage = [self.function.storage(zeta + p, p) for zeta in zetas]
            quad_storage = [quad(self.function, -p, zeta)[0] for zeta in zetas]
            assert_close(storage, quad_storage)

    def test_shallow_storage(self):
        """Verify computation of shallow storage

        and that storage = shallow_storage + deep_storage

        """
        zetas = np.linspace(-300, 300, 100)
        surface = np.linspace(10.0, 200.0, 20)
        shallow_storage = np.array(
            [self.function.shallow_storage(zeta) for zeta in zetas]
        )
        for p in surface:
            storage = np.array([self.function.storage(zeta + p, p) for zeta in zetas])
            deep_storage = self.function.deep_storage(p)
            assert_close(
                storage,
                deep_storage + shallow_storage,
                'storage = deep_storage + shallow_storage',
            )
            zetas_too = [
                self.function.zeta(storage[i] - deep_storage)
                for i in range(len(storage))
            ]
            assert_close(zetas, zetas_too, 'round-trip of zeta')

    def test_zeta(self):
        """Verify that the zeta() is the inverse of shallow_storage"""
        zetas = np.linspace(-600, 600, 100)
        shallow_storage = [self.function.shallow_storage(zeta) for zeta in zetas]
        zetas_too = [self.function.zeta(delta_W) for delta_W in shallow_storage]
        assert_close(zetas, zetas_too)
        assert_close(
            self.function.shallow_storage(0), 0, 'delta_W(0) == 0 by definition'
        )
        assert_close(self.function.zeta(0), 0, 'zeta(0) == 0 by definition')


class TestSpecificYieldToo(TestSpecificYield):
    """More test code for specific yield calculation class"""

    zeta_knots = [
        -291.661823185,
        -183.116244185,
        -15.7362692649,
        10.6471477151,
        38.7848923151,
        168.338176815,
    ]
    Sy_knots = [
        0.1003017029684028,
        0.1914710774835498,
        0.2451087076108037,
        0.3059738850847126,
        0.3007979508742346,
        0.689328814885657,
    ]

    def test_inversion(self):
        """Verify stable inversion between zeta and shallow_storage"""
        zetas = np.linspace(-600, 600, 100)
        shallow_storage = [self.function.shallow_storage(zeta) for zeta in zetas]
        zetas_too = [self.function.zeta(delta_W) for delta_W in shallow_storage]
        assert_close(zetas, zetas_too)
        assert_close(
            self.function.shallow_storage(0), 0, 'delta_W(0) == 0 by definition'
        )
        assert_close(self.function.zeta(0), 0, 'zeta(0) == 0 by definition')
        for zeta in zetas:
            zeta_ref = zeta
            for i in range(500):
                Ws = self.function.shallow_storage(zeta)
                zeta = self.function.zeta(Ws)
                assert_close(
                    zeta,
                    zeta_ref,
                    'Zeta {} not close to {} after {} iterations'.format(
                        zeta, zeta_ref, i
                    ),
                )
