"""Test code for command-line interface"""

import os
import tempfile

import pytest

from spowtd.test import conftest
import spowtd.user_interface as cli_mod


def test_get_version(capfd):
    """Invoking spowtd --version returns version and exits with 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['--version'])
    assert exception.type == SystemExit
    assert exception.value.code == 0
    out, _ = capfd.readouterr()
    with open(
        os.path.join(os.path.dirname(__file__), os.pardir, 'VERSION.txt'),
        'rt',
        encoding='utf-8',
    ) as version_file:
        version = version_file.read()
    assert out == version


def test_help():
    """Invoking spowtd --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_load_help():
    """Invoking spowtd load --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['load', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_load():
    """'spowtd load' exits without error"""
    paths = {
        key: os.path.join(conftest.SAMPLE_DATA_DIR, f'{key}_1.txt')
        for key in ('evapotranspiration', 'precipitation', 'water_level')
    }
    with tempfile.NamedTemporaryFile(
        suffix='.sqlite3', delete=False, delete_on_close=True
    ) as db_file:
        cli_mod.main(
            [
                'load',
                db_file.name,
                '--precipitation',
                paths['precipitation'],
                '--evapotranspiration',
                paths['evapotranspiration'],
                '--water-level',
                paths['water_level'],
                '--timezone',
                'Africa/Lagos',
            ]
        )


def test_classify_help():
    """Invoking spowtd classify --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['classify', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_plot_help():
    """Invoking spowtd plot --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['plot', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_plot_specific_yield_help():
    """Invoking spowtd plot specific-yield --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['plot', 'specific-yield', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_plot_conductivity_help():
    """Invoking spowtd plot specific-yield --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['plot', 'conductivity', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_plot_transmissivity_help():
    """Invoking spowtd plot specific-yield --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['plot', 'transmissivity', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_rise_help():
    """Invoking spowtd rise --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['rise', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_simulate_help():
    """Invoking spowtd simulate --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['simulate', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_simulate_rise_help():
    """Invoking spowtd simulate rise --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['simulate', 'rise', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_set_curvature_help():
    """Invoking spowtd simulate rise --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['set-curvature', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_simulate_recession_help():
    """Invoking spowtd simulate recession --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['simulate', 'recession', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_pestfiles_help():
    """Invoking spowtd pestfiles --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['pestfiles', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_pestfiles_rise_help():
    """Invoking spowtd pestfiles rise --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['pestfiles', 'rise', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0


def test_pestfiles_curves_help():
    """Invoking spowtd pestfiles curves --help exits with code 0"""
    with pytest.raises(SystemExit) as exception:
        cli_mod.main(['pestfiles', 'curves', '--help'])
    assert exception.type == SystemExit
    assert exception.value.code == 0
