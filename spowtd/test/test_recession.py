"""Test code for recession module

"""

import spowtd.recession as recession_mod


def test_recession(classified_connection):
    """Assemble recession curve

    """
    recession_mod.find_recession_offsets(
        classified_connection,
        delta_z_mm=1.0)
