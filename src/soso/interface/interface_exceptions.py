'''
Definitions of exceptions that can occur when interacting with the scheduler's
interface.
'''


class MissingInputHashException(Exception):
    '''
    Exception raised when the scheduler was requested to re-run, but an input
    hash was not provided, meaning the scheduler was not provided with a
    previous schedule to operate on.
    '''

    def __init__(self):
        super().__init__(
            'Rescheduling was run without reference to existing scheduling '
                'output'
        )


class ExistingScheduleNotFoundException(Exception):
    '''
    Exception raised when the scheduler was requested to re-run with an input
    hash to an existing schedule, but the existing schedule being referenced was
    not found.
    '''

    def __init__(self):
        super().__init__(
            'Existing scheduling output was referenced in rescheduling '
                'operation, but the schedule was not found'
        )
