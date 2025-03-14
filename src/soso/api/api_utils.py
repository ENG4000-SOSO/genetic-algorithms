'''
Utility functions for API (and FastAPI) operations.
'''


from fastapi import Request


def get_client_ip_and_port(request: Request):
    '''
    Returns the client's host (IP) and port.

    Args:
        request: The request from the client.

    Returns:
        A tuple containing the client's host and port, both as strings.
    '''

    if request.client:
        return request.client.host, str(request.client.port)
    else:
        return '?', '?'
