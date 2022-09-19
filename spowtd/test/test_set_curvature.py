"""Test code for setting site curvature

"""

import spowtd.set_curvature as set_curvature_mod


def test_set_curvature(classified_connection):
    """Set site curvature"""
    ref_curvature = 9.876
    set_curvature_mod.set_curvature(classified_connection, ref_curvature)
    cursor = classified_connection.cursor()
    cursor.execute("SELECT curvature_m_km2 FROM curvature")
    curvature_m_km2 = cursor.fetchone()[0]
    assert curvature_m_km2 == ref_curvature
    cursor.close()
