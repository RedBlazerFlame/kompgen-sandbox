"""Generates DETERMINISTIC tests."""

from sys import *
from kg.generators import * ### @import
from formatter import * ### @import

@listify
def gen_deterministic(rand, *args):
    ... # write your generator here

    # example:
    x, y = map(int, args[:2])

    return x, y


if __name__ == '__main__':
    write_to_file(format_case, gen_deterministic, argv[1:], stdout)
