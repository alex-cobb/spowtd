"""Test code for Spowtd

"""


def set_dataset_id(connection, dataset_id):
    """Set dataset id

    Creates a singleton table to carry through the sample as a dataset id. This
    dataset id is then used as a key for expected results in other tests.
    """
    cursor = connection.cursor()
    cursor.execute(
        """
    CREATE TABLE dataset (
      id integer NOT NULL PRIMARY KEY,
      valid integer NOT NULL UNIQUE DEFAULT 1
        CHECK (valid = 1)
    )"""
    )
    cursor.execute("INSERT INTO dataset (id) VALUES (?)", (dataset_id,))
    cursor.close()
    connection.commit()


def get_dataset_id(connection):
    """Retrieve dataset id from connection

    See set_dataset_id for more.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM dataset")
    dataset_id = cursor.fetchone()[0]
    cursor.close()
    return dataset_id
