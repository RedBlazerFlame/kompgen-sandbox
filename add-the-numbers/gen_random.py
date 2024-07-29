"""Generates RANDOM tests."""

from sys import *
from kg.generators import * ### @import
from formatter import * ### @import

@listify
def gen_random(rand, *args):
    ... # write your generator here

    # example:
    x, y = map(int, args[:2])

    return rand.randint(-x, x), rand.randint(-y, y)


if __name__ == '__main__':
    write_to_file(format_case, gen_random, argv[1:], stdout)
