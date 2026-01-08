"""Utilities for tests

"""

import numpy as np


def assert_close(a, b, message='', rtol=1e-5, atol=1e-8):
    """Verify that floats in a and b are close

    """
    message_template = '{} not close to {}'
    if message:
        message_template = ': '.join((message, message_template))
    assert np.allclose(a, b, rtol=rtol, atol=atol), \
        message_template.format(a, b)
