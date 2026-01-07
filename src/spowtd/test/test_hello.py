"""Smoke test for Cython prototype"""


def test_hello():
    """Smoke test for Cython prototype"""
    import spowtd.hello

    assert spowtd.hello.add(3, 4) == 7
