"""Test code for Spowtd"""


def get_dataset_id(request):
    """Retrieve dataset id from request fixture

    Obtains dataset id as the parameter to the persistent_loaded_connection
    fixture.
    """
    return request.node.callspec.params['persistent_loaded_connection']
