"""Test code for data loading module

"""

import os
import sqlite3

import spowtd.load as load_mod


SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    'sample_data')


def test_sample_data_readable():
    """Sample data files are present and readable in text mode

    """
    for file_type in ('evapotranspiration',
                      'precipitation',
                      'water_level'):
        for sample in (1, 2):
            data_path = get_sample_file_path(file_type, sample)
            assert os.path.exists(data_path)
            with open(data_path, 'rt') as data_file:
                data_file.read()


def test_load_sample_data():
    """Spowtd loads sample data without errors

    """
    sample = 1
    with sqlite3.connect(':memory:') as connection:
        load_mod.load_data(
            connection=connection,
            precipitation_data_file=open(
                get_sample_file_path(
                    'precipitation', sample), 'rt'),
            evapotranspiration_data_file=open(
                get_sample_file_path(
                    'evapotranspiration', sample), 'rt'),
            water_level_data_file=open(
                get_sample_file_path(
                    'water_level', sample), 'rt'))


def get_sample_file_path(file_type, sample):
    """Return path to a sample file

    """
    return os.path.join(SAMPLE_DATA_DIR,
                        '{}_{}.txt'.format(file_type, sample))
