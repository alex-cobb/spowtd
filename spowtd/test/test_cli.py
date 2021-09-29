"""Test code for command-line interface

"""

import os

import pytest

import spowtd.user_interface as cli_mod


def test_get_version(capfd):
    """Invoking spowtd --version returns version and exits with 0

    """
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['--version'])
    assert exception.type == SystemExit
    assert exception.value.code == 0
    out, _ = capfd.readouterr()
    with open(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                'VERSION.txt'), 'rt') as version_file:
        version = version_file.read()
    assert out == version


def test_load_help():
    """Invoking spowtd load --help exits with code 0

    """
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['load', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0
