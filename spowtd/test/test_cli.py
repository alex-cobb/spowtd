"""Test code for command-line interface

"""

import os
import tempfile

import pytest

from spowtd.test import conftest
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


def test_help():
    """Invoking spowtd --help exits with code 0

    """
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_load_help():
    """Invoking spowtd load --help exits with code 0

    """
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['load', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_load():
    """'spowtd load' exits without error

    """
    paths = {
        key: os.path.join(
            conftest.SAMPLE_DATA_DIR,
            '{}_1.txt'.format(key))
        for key in ('evapotranspiration',
                    'precipitation',
                    'water_level')}
    with tempfile.NamedTemporaryFile(
            suffix='.sqlite3') as db_file:
        cli_mod.main([
            'load',
            db_file.name,
            '--precipitation', paths['precipitation'],
            '--evapotranspiration', paths['evapotranspiration'],
            '--water-level', paths['water_level'],
            '--timezone', 'Africa/Lagos'])


def test_classify_help():
    """Invoking spowtd classify --help exits with code 0

    """
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['classify', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0
