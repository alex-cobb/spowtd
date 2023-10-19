"""Test code for rise module

"""

import spowtd.rise as rise_mod


def test_rise(classified_connection):
    """Assemble rising curve"""
    rise_mod.find_rise_offsets(classified_connection)


def test_rise_covariance(classified_connection):
    """Assemble covariance of rise event errors"""
    rise_mod.get_rise_covariance(classified_connection)
