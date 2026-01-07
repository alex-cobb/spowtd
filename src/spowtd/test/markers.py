"""Markers for tests"""

import sys

import pytest

# For Linux-only tests
linux_only = pytest.mark.skipif(sys.platform != 'linux', reason='Linux only')
