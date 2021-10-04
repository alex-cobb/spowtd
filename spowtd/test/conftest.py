"""Fixtures for spowtd tests

"""

import os
import sqlite3

import spowtd.classify as classify_mod
import spowtd.load as load_mod
import spowtd.zeta_grid as zeta_grid_mod

import pytest


SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    'sample_data')


collect_ignore = []  # pylint: disable=invalid-name

for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
    collect_ignore += [
        os.path.join(dirpath, filename.replace('.py', '_flymake.py'))
        for filename in filenames if filename.endswith('.py')]


# pylint: disable=redefined-outer-name
@pytest.fixture
def connection():
    """Connection to an in-memory database

    """
    with sqlite3.connect(':memory:') as in_memory_db:
        yield in_memory_db


@pytest.fixture(scope="function", params=[1, 2])
def loaded_connection(request, connection):
    """Connection to in-memory database with loaded data

    """
    sample = request.param
    load_mod.load_data(
        connection=connection,
        precipitation_data_file=open(
            get_sample_file_path(
                'precipitation', sample),
            'rt', encoding='utf-8-sig'),
        evapotranspiration_data_file=open(
            get_sample_file_path(
                'evapotranspiration', sample),
            'rt', encoding='utf-8-sig'),
        water_level_data_file=open(
            get_sample_file_path(
                'water_level', sample),
            'rt', encoding='utf-8-sig'))
    yield connection


@pytest.fixture(scope="function")
def classified_connection(loaded_connection):
    """Connection to in-memory database with classified data

    """
    classify_mod.classify_intervals(
        loaded_connection,
        storm_rain_threshold_mm_h=4.0,
        rising_jump_threshold_mm_h=8.0)
    zeta_grid_mod.populate_zeta_grid(
        loaded_connection,
        grid_interval_mm=1.0)
    yield loaded_connection


def get_sample_file_path(file_type, sample):
    """Return path to a sample file

    """
    return os.path.join(SAMPLE_DATA_DIR,
                        '{}_{}.txt'.format(file_type, sample))
