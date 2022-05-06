"""Setting site curvature in database

"""

def set_curvature(connection, curvature_m_km2):
    """Set site curvature

    """
    cursor = connection.cursor()
    cursor.execute("""
    INSERT INTO curvature (curvature_m_km2)
    VALUES (?)""", (curvature_m_km2,))
    cursor.close()
