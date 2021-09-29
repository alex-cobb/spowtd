"""Load data into Spowtd data file

"""

import os
import sqlite3


SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
    'schema.sql')


def load_data(connection,
              precipitation_data_file,
              evapotranspiration_data_file,
              water_level_data_file):
    """Load data into Spowtd data file

    """
    connection.execute("PRAGMA foreign_keys = 1")
    cursor = connection.cursor()
    with open(SCHEMA_PATH, 'rt') as schema_file:
        cursor.executescript(schema_file.read())
    cursor.close()
    # XXX
    for line in precipitation_data_file:
        print(line.strip())
    for line in evapotranspiration_data_file:
        print(line.strip())
    for line in water_level_data_file:
        print(line.strip())
