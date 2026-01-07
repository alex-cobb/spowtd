# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Markers for tests"""

import sys

import pytest

# For Linux-only tests
linux_only = pytest.mark.skipif(sys.platform != 'linux', reason='Linux only')
