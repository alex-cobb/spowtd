# Copyright (C) 2026 Alex Cobb
# Licensed under the BSD 2-Clause License (see LICENSE-BSD.txt)
#
# NOTE: This file is part of the spowtd project. When distributed as a compiled binary
# linked against the GSL, the resulting work is licensed under the GPL-3.0-or-later.

"""Fixtures for spowtd tests"""

import os
import sqlite3
import subprocess

import pytest

import spowtd.classify as classify_mod
import spowtd.load as load_mod
import spowtd.zeta_grid as zeta_grid_mod


SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')


collect_ignore = []  # pylint: disable=invalid-name

for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
    collect_ignore += [
        os.path.join(dirpath, filename.replace('.py', '_flymake.py'))
        for filename in filenames
        if filename.endswith('.py')
    ]


# pylint: disable=redefined-outer-name
@pytest.fixture(scope='session', params=[1, 2])
def persistent_loaded_connection(request):
    """Connection to in-memory database with loaded data

    Each loaded connection is created once per test session by loading data from a set
    of sample files (indexed by param).  Tests work with copies.

    """
    with sqlite3.connect(':memory:') as persistent_connection:
        sample = request.param
        with (
            open(
                get_sample_file_path('precipitation', sample),
                'rt',
                encoding='utf-8-sig',
            ) as precip_f,
            open(
                get_sample_file_path('evapotranspiration', sample),
                'rt',
                encoding='utf-8-sig',
            ) as et_f,
            open(
                get_sample_file_path('water_level', sample),
                'rt',
                encoding='utf-8-sig',
            ) as zeta_f,
        ):
            load_mod.load_data(
                connection=persistent_connection,
                precipitation_data_file=precip_f,
                evapotranspiration_data_file=et_f,
                water_level_data_file=zeta_f,
                time_zone_name='Africa/Lagos',
            )
            yield persistent_connection


@pytest.fixture(scope='function')
def connection():
    """Connection to an in-memory database"""
    with sqlite3.connect(':memory:') as in_memory_db:
        yield in_memory_db


@pytest.fixture(scope='function')
def loaded_connection(connection, persistent_loaded_connection):
    """Connection to in-memory database with loaded data"""
    persistent_loaded_connection.backup(connection)
    yield connection


@pytest.fixture(scope='function')
def classified_connection(loaded_connection):
    """Connection to in-memory database with classified data"""
    classify_mod.classify_intervals(
        loaded_connection,
        storm_rain_threshold_mm_h=8.0,
        rising_jump_threshold_mm_h=5.0,
    )
    zeta_grid_mod.populate_zeta_grid(loaded_connection, grid_interval_mm=1.0)
    yield loaded_connection


def get_sample_file_path(file_type, sample, suffix='txt'):
    """Return path to a sample file"""
    return os.path.join(SAMPLE_DATA_DIR, f'{file_type}_{sample}.{suffix}')


def get_parameter_file_path(file_type):
    """Return path to a parameter file"""
    return os.path.join(SAMPLE_DATA_DIR, f'{file_type}_parameters.yml')


def peatclsm_specific_yield_table():
    """Return a table of specific yields as parameterized in PEATCLSM"""
    return get_peatclsm_data_table(
        'specific_yield', ['"water_level_m"', '"specific_yield"']
    )


def peatclsm_transmissivity_table():
    """Return a table of transmissivities as parameterized in PEATCLSM"""
    return get_peatclsm_data_table(
        'transmissivity', ['"water_level_m"', '"transmissivity_m2_s"']
    )


def get_peatclsm_data_table(variable, expected_header):
    """Return a tabulated variable as parameterized in PEATCLSM"""
    output = subprocess.check_output(
        [
            'Rscript',
            '--vanilla',
            os.path.join(os.path.dirname(__file__), 'peatclsm_hydraulic_functions.R'),
            variable,
        ],
        encoding='utf-8',
    )
    rows = [line.strip().split(',') for line in output.splitlines()]
    assert rows.pop(0) == expected_header
    return [tuple(float(value) for value in row) for row in rows]
