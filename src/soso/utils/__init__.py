'''
Utilities to be used throughout the SOSO project.
'''

def counter_generator():
    '''
    Generates a counter to make counting easier and more Pythonic.

    To initialize the counter:

    ```
    counter = counter_generator()
    ```

    then, increment the counter:

    ```
    next(counter)
    ```
    '''
    i = 0
    while True:
        yield i
        i += 1
