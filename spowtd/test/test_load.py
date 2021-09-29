"""Test code for data loading module

"""

import os
import sqlite3

import spowtd.load as load_mod


TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    'sample_data')


def test_sample_data_exists():
    """Sample data files are present

    """
    for basename in ['{}_{}.txt'.format(file_type, sample)
                     for file_type in ('evapotranspiration',
                                       'precipitation',
                                       'water_level')
                     for sample in (1, 2)]:
        assert os.path.exists(
            os.path.join(TEST_DATA_DIR,
                         basename))


def test_load_sample_data():
    """Spowtd loads sample data without errors

    """
    with sqlite3.connect(':memory:') as connection:
        load_mod.load_data(connection=connection)
