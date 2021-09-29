"""Load data into Spowtd data file

"""

import os
import sqlite3


SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
    'schema.sql')


def load_data(connection):
    """Load data into Spowtd data file

    """
    connection.execute("PRAGMA foreign_keys = 1")
    cursor = connection.cursor()
    with open(SCHEMA_PATH, 'rt') as schema_file:
        cursor.executescript(schema_file.read())
    cursor.close()
