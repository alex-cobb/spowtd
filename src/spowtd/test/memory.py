# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Shared code for memory tests of extension modules"""

import resource
import sys


def get_memory_usage_kb():
    """Get memory usage of the current process in kilobytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def repeat_create_destroy(factory, refcounts, *args, **kwargs):
    """Making objects does not increase memory or ref counts"""
    iterations = 10000
    # Is this always a valid ceiling?
    max_memory_increase = 1000

    base_memory_kb = get_memory_usage_kb()

    for i in range(iterations):
        obj = factory(*args, **kwargs)
        assert sys.getrefcount(obj) == 1, sys.getrefcount(obj)
        for attrname, refcount in refcounts.items():
            assert sys.getrefcount(getattr(obj, attrname)) == refcount
        memory_usage_kb = get_memory_usage_kb()
        assert memory_usage_kb - base_memory_kb <= max_memory_increase, (
            '{} kb > {} kb at iteration {}'.format(
                memory_usage_kb - base_memory_kb, max_memory_increase, i
            )
        )
