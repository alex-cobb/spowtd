# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Test code for Spowtd"""


def get_dataset_id(request):
    """Retrieve dataset id from request fixture

    Obtains dataset id as the parameter to the persistent_loaded_connection fixture.

    """
    return request.node.callspec.params['persistent_loaded_connection']
