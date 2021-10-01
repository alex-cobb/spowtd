"""Test code for interval classification module

"""

import spowtd.classify as classify_mod


def test_classify(loaded_connection):
    """Classify data into storm and interstorm intervals

    """
    classify_mod.classify_intervals(
        loaded_connection,
        grid_interval_mm=1.0,
        storm_rain_threshold_mm_h=4.0,
        rising_jump_threshold_mm_h=8.0)
