"""Checks whether the input file is valid."""

from sys import *
from kg.validators import * ### @import

bounds = {
    'x': -10**18 <= +Var <= 10**18,
    'y': -10**18 <= +Var <= 10**18,
}

subtasks = {
    '1': {
        'x': 0 <= +Var <= 1,
        'y': 0 <= +Var <= 1,
    },
    '2': {
        'x': -1 <= +Var <= 1,
        'y': -1 <= +Var <= 1,
    },
    '3': {
        'x': -1000 <= +Var <= 1000,
        'y': -1000 <= +Var <= 1000,
    },
    '4': {
        'x': -10**6 <= +Var <= 10**6,
        'y': -10**6 <= +Var <= 10**6,
    },
    '5': {},
}

@validator(bounds=bounds, subtasks=subtasks)
def validate(stream, subtask=None, *, lim):
    [x, y] = stream.read.int(lim.x).space.int(lim.y).eoln
    [] = stream.read.eof

    # other possibilities
    # [x, y, z] = stream.read.real(lim.x).space.real(lim.y).space.int(lim.z).eoln
    # [line] = stream.read.line(lim.s).eoln
    # [name] = stream.read.token(lim.name).eoln


if __name__ == '__main__':
    validate_or_detect_subtasks(validate, subtasks, stdin)
