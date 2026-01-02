"""Test code for data loading module"""


def test_load_sample_data(loaded_connection):
    """Spowtd correctly loads sample data"""
    cursor = loaded_connection.cursor()
    # XXX Check loaded data
    cursor.close()
