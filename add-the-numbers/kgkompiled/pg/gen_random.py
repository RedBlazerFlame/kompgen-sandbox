#!/usr/bin/env python3
"""Generates RANDOM tests."""

from sys import *
#BLACKMAGIC start import kg.generators (as kg.generators)
BM0, __name__ = __name__, "BM0"
import functools, io, random, sys

#BLACKMAGIC start import kg.utils (as .utils)
BM1, __name__ = __name__, "BM1"
#BLACKMAGIC start import kg.utils.utils (as .utils)
BM2, __name__ = __name__, "BM2"
import collections, collections.abc, functools, os, os.path, pathlib, re, sys

warn_print = print

def noop(*a, **kw): ...


def warn_print(*a, **kw):
    if not warn_print.wp:
        try:
            from kg.script.utils import warn_print as _wp
        except ImportError:
            warn_print.wp = print
        else:
            warn_print.wp = _wp

    return warn_print.wp(*a, **kw)
warn_print.wp = None

def warn(warning):
    ...


CURR_PLATFORM = 'pg' 



def abs_error(a, b):
    return abs(a - b)

def abs_rel_error(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1)

def overflow_ell(s, ct=50, etc='...'):
    assert len(etc) <= ct
    s = str(s)
    return s if len(s) <= ct else s[-len(etc):] + etc

def ensure(condition, message="ensure condition failed. (see Traceback to determine which one)", exc=Exception):
    ''' assert that doesn't raise AssertionError. Useful/Convenient for judging. '''
    if not condition:
        try:
            message = message()
        except TypeError:
            ...
        if isinstance(message, str):
            message = exc(message)
        raise message or exc


def apply_after(g, name=None):
    ''' Make a decorator that applies "g" to the return value of a function. '''
    def _d(f):
        @functools.wraps(f)
        def _f(*args, **kwargs):
            return g(f(*args, **kwargs))
        return _f
    if name is not None: _d.__name__ = name
    return _d

listify = apply_after(list, 'listify')

memoize = functools.lru_cache(maxsize=None)

t_inf = 10**18
r_int = r'0|(?:-?[1-9]\d*)'
r_sint = r'[+-](?:0|(?:[1-9]\d*))'

t_patterns = [re.compile(rf'^{pat}\Z') for pat in [
    rf'(?P<start>{r_int})(?P<range>(?:\.\.)|-)(?P<end>{r_int})\((?P<step>{r_sint})\)',
    rf'(?P<start>{r_int})(?P<range>(?:\.\.)|-)(?P<end>{r_int})',
    rf'(?P<start>{r_int})(?P<range>\.\.)\((?P<step>{r_sint})\)',
    rf'(?P<start>{r_int})(?P<range>\.\.)',
    rf'(?P<start>{r_int})',
]]

def _t_range_args(s, *, inf=t_inf, patterns=t_patterns):
    for pat in patterns:
        m = pat.match(s)
        if m:
            m = m.groupdict()
            start = int(m['start'])
            if m.get('range'):
                step = int(m.get('step', 1))
                if 'end' in m:
                    end = int(m['end'])
                    if step < 0: end -= 1
                    if step > 0: end += 1
                else:
                    if step < 0:
                        end = -inf
                    elif step > 0:
                        end = +inf
                    else:
                        end = None
            else:
                step = 1
                end = start + 1
            if step and end is not None and (end - start) * step >= 0:
                return start, end, step
    raise ValueError(f"Range cannot be read: {s!r}")

def t_range(r, *, inf=t_inf):
    return range(*_t_range_args(r, inf=inf))

def t_infinite(r, *, inf=t_inf):
    start, end, step = _t_range_args(r, inf=inf)
    return abs(end) >= inf

def t_sequence_ranges(s):
    return [t_range(p) for p in s.split(',')]

def t_sequence(s):
    for r in s.split(','):
        yield from t_range(r)

@listify
def list_t_sequence(s):
    for r in s.split(','):
        if t_infinite(r):
            raise ValueError(f"Cannot form a list from an infinite range {r}")
        yield from t_range(r)

def compress_t_sequence(s, *, inf=t_inf):
    def exactize(start, end, step):
        return start, end + (start - end) % step, step
    def decode(r):
        return exactize(*_t_range_args(r, inf=inf))
    @listify
    def combine_ranges(a, b):
        astart, aend, astep = a
        bstart, bend, bstep = b
        if astep == bstep and aend == bstart:
            yield astart, bend, astep
        else:
            yield from (a, b)
    def merge_ranges(ranges1, ranges2):
        *ranges1, erange1 = ranges1
        frange2, *ranges2 = ranges2
        return list(ranges1) + combine_ranges(erange1, frange2) + list(ranges2)
    def encode(start, end, step):
        assert (end - start) % step == 0
        end = '' if abs(end) >= inf else end - step
        if end != '': assert (end - start) * step >= 0
        if start == end:
            return str(start)
        else:
            assert step
            step_sgn = '+' if step > 0 else '-' if step < 0 else '?'
            step_str = '' if step == 1 else f'({step_sgn}{abs(step)})'
            range_ = '..' if end == '' else '-'
            return f'{start}{range_}{end}{step_str}'
    return ','.join(encode(*t) for t in functools.reduce(merge_ranges, ([decode(r)] for r in s.split(','))))


def file_sequence(s, *, mktemp=False):
    if s.startswith(':'):
        if mktemp:
            pathlib.Path('temp').mkdir(parents=True, exist_ok=True)
        for v in t_sequence(s[1:]):
            yield os.path.join('temp', str(v))
    else:
        yield from map(str, t_sequence(s))


def default_return(ret):
    def _d(f):
        @functools.wraps(f)
        def _f(*args, **kwargs):
            res = f(*args, **kwargs)
            return res if res is not None else ret
        return _f
    return _d

default_score = default_return(1.0)



EOF = ''
EOLN = '\n'
SPACE = ' '

def stream_char_label(ch):
    if ch == EOF: return 'end-of-file'
    assert len(ch) == 1
    return repr(ch)

def force_to_set(s):
    if not isinstance(s, collections.abc.Set):
        s = frozenset(s)
    return s




class Builder:
    def __init__(self, name, build_standalone, build_from_parts):
        self.name = name
        self.build_standalone = build_standalone
        self.build_from_parts = build_from_parts
        self.pending = None
        super().__init__()

    def start_building(self):
        if self.pending is None: self.pending = self.build_from_parts()
        return self.pending

    def set(self, arg):
        self.start_building()
        if callable(arg):
            try:
                name = arg.__name__
            except AttributeError:
                ...
            else:
                return self._set(name, arg)

        return functools.partial(self._set, arg)

    def _set(self, name, arg):
        self.start_building()
        return self.pending._set(name, arg)

    def make(self, *args, **kwargs):
        if self.pending is None: raise RuntimeError("Cannot build: no .set done. Did you mean @checker?")

        for name in self.pending._names:
            if name in kwargs: self._set(name, kwargs.pop(name))

        interact = self.pending
        self.pending = None
        interact.init(*args, **kwargs)
        return interact

    def __call__(self, *args, **kwargs):
        if self.pending is not None: raise RuntimeError(f"Cannot build standalone {self.name} if .set has been done. Did you mean to call .make?")
        return self.build_standalone(*args, **kwargs)



def warn_on_call(warning):
    _d = lambda f: f
    return _d

def deprec_name_warning(*a, **kw): return '!'
warn_deprec_name = noop
def deprec_alias(oname, nobj, *a, **kw): return nobj



class ChainRead:
    def __init__(self, stream):
        self._s = stream
        self._r = collections.deque()
        super().__init__()

    def __iter__(self):
        while self._r:
            yield self._r.popleft()

    def __call__(self): return list(self)


    def line(self, *a, **kw):
        self._r.append(self._s.read_line(*a, **kw)); return self

    def int(self, *a, **kw):
        self._r.append(self._s.read_int(*a, **kw)); return self

    def ints(self, *a, **kw):
        self._r.append(self._s.read_ints(*a, **kw)); return self

    def real(self, *a, **kw):
        self._r.append(self._s.read_real(*a, **kw)); return self

    def reals(self, *a, **kw):
        self._r.append(self._s.read_reals(*a, **kw)); return self

    def token(self, *a, **kw):
        self._r.append(self._s.read_token(*a, **kw)); return self

    def tokens(self, *a, **kw):
        self._r.append(self._s.read_tokens(*a, **kw)); return self

    def until(self, *a, **kw):
        self._r.append(self._s.read_until(*a, **kw)); return self

    def while_(self, *a, **kw):
        self._r.append(self._s.read_while(*a, **kw)); return self

    def char(self, *a, **kw):
        res = self._s.read_char(*a, **kw)
        if res is not None: self._r.append(res)
        return self

    @property
    def space(self):
        self._s.read_space(); return self

    @property
    def eoln(self):
        self._s.read_eoln(); return self

    @property
    def eof(self):
        self._s.read_eof(); return self

    @property
    def spaces(self):
        self._s.read_spaces(); return self


def pop_callable(s):
    f = None
    if len(s) >= 1 and callable(s[0]): f, *s = s
    return f, s


__name__ = BM2
del BM2
#BLACKMAGIC end import kg.utils.utils (as .utils)
__name__ = BM1
del BM1
#BLACKMAGIC end import kg.utils (as .utils)

class GeneratorError(Exception): ...

@listify
def group_into(v, seq):
    buf = []
    for s in seq:
        buf.append(s)
        if len(buf) > v: raise ValueError("v cannot be zero if seq is nonempty")
        if len(buf) == v:
            yield buf
            buf = []
    if buf: yield buf


class KGRandom(random.Random):
    def shuffled(self, x):
        x = list(x)
        self.shuffle(x)
        return x
    shuff = shuffled

    def randinterval(self, a, b):
        while True:
            x = self.randint(a, b)
            y = self.randint(a, b)
            if x <= y: return x, y

    def randmerge(self, *x):
        # divide and conquer for speed
        if not x: return []
        if len(x) == 1: return list(x[0])
        return self.randmerge2(self.randmerge(*x[::2]), self.randmerge(*x[1::2]))

    def randmerge2(self, a, b):
        a = list(a)[::-1]
        b = list(b)[::-1]
        res = []
        while a or b:
            res.append((a if self.randrange(len(a) + len(b)) < len(a) else b).pop())
        return res

    def randdistrib(self, total, count, *, min_=0, max_=None, skew=1): 
        if min_*count > total:
            raise ValueError(f"The total must be at least {min_}*{count}={min_*count} when count={count} and min_={min_}")
        if max_ is not None and max_*count < total:
            raise ValueError(f"The total must be at most {max_}*{count}={max_*count} when count={count} and max_={max_}")
        if skew <= 0:
            raise ValueError("The skew has to be at least 1.")
        if max_ is None:
            max_ = total
        dist = [min_]*count

        inds = self.shuffled(range(count))
        for it in range(total - min_*count):
            while True:
                assert inds
                idx = min(self.randrange(len(inds)) for it in range(skew))
                if dist[inds[idx]] < max_:
                    dist[inds[idx]] += 1
                    break
                else:
                    inds[idx], inds[-1] = inds[-1], inds[idx]
                    inds.pop()

        assert sum(dist) == total
        assert min_ <= min(dist) <= max(dist) <= max_

        return dist

    @listify
    def randpartition(self, total, min_=1, skew=2): 
        if total < 0: raise ValueError("The total should be at least 0.")
        if min_ <= 0: raise ValueError("The value of min_ should be at least 1.")
        if skew <= 0: raise ValueError("The skew should be at least 1.")
        if total == 0:
            return []

        it = 0
        for i in range(total - min_):
            it += 1
            if it >= min_ and not self.randrange(skew):
                yield it
                it = 0
        yield it + min_



_pmod = 2013265921
_pbase = 1340157138
_xmod = 10**9 + 7
_xbase = 790790578
_xor = 0xDEAFBEEFEE
def _chash_seq(seq, *, _pmod=_pmod, _pbase=_pbase, _xmod=_xmod, _xor=_xor):
    pol = 0
    xol = 0
    for s in seq:
        pol = (pol * _pbase + s) % _pmod
        xol = ((xol * _xbase + s) ^ _xor) % _xmod
    return (pol << 32) ^ xol


def _make_seed(args):
    return _chash_seq(_chash_seq(map(ord, arg)) for arg in args) ^ 0xBEABDEEF


def _write_with_validate(format_case, file, case, *, validate=None):
    if validate is not None:
        tfile = io.StringIO()
        format_case(tfile, case)
        tfile.seek(0) # reset the StringIO
        validate(tfile)
        file.write(tfile.getvalue())
    else:
        format_case(file, case)


class DistribCase:
    def __init__(self, make, distribute, *, single_case=False):
        self.make = make
        self.distribute = distribute
        self.single_case = single_case
        super().__init__()

    def lazy(self, rand, *args):
        casemakers = []
        def mnew_case(*fwd_args, **info):
            def _mnew_case(f):
                nrand_seed = rand.getrandbits(64) ^ 0xC0BFEFE
                @functools.wraps(f)
                def new_f():
                    return f(KGRandom(nrand_seed), *fwd_args)
                casemakers.append(new_f)
                mnew_case.total_cases += 1
                for name, value in info.items():
                    setattr(new_f, name, value)
            return _mnew_case
        mnew_case.total_cases = 0
        self.make(rand, mnew_case, *args)

        def dnew_case(*fwd_args, **info):
            def _dnew_case(f):
                nrand_seed = rand.getrandbits(64) ^ 0xC0BFEFE
                @functools.wraps(f)
                def new_f():
                    return f(KGRandom(nrand_seed), *fwd_args)
                for name, value in info.items(): # forward any info
                    setattr(new_f, name, value)
                return new_f
            return _dnew_case
        return self.distribute(rand, dnew_case, casemakers, *args)

    def __call__(self, rand, *args):
        return map(self.realize, self.lazy(rand, *args))

    def realize(self, group):
        return group() if self.single_case else [make() for make in group]

    def __getitem__(self, index):
        def get(rand, *args):
            groups = self.lazy(rand, *args)
            if not (0 <= index < len(groups)): raise GeneratorError(f"Invalid index: {index} of {len(groups)} groups")
            return self.realize(groups[index])
        return get

def write_to_file(format_case, make, args, file, *, validate=None): 
    try:
        make, distribute, index = make
    except (ValueError, TypeError):
        ...
    else:
        make = DistribCase(make, distribute)[index]

    rand = KGRandom(_make_seed(args))
    case = make(rand, *args)
    _write_with_validate(format_case, file, case, validate=validate)


def write_to_files(format_case, make, filenames, *args, validate=None):
    try:
        make, distribute = make
    except (ValueError, TypeError):
        ...
    else:
        make = DistribCase(make, distribute)

    rand = KGRandom(_make_seed(args))

    if filenames == "COUNT":
        print(sum(1 for case in make(rand, *args)))
        return

    if isinstance(filenames, str):
        filenames = file_sequence(filenames, mktemp=True)
    filenames = iter(filenames)
    filecount = 0
    for index, case in enumerate(make(rand, *args)):
        try:
            filename = next(filenames)
        except StopIteration as st:
            raise GeneratorError(f"Not enough files! Need more than {index}") from st
        with open(filename, 'w') as file:
            _write_with_validate(format_case, file, case, validate=validate)
        filecount += 1
__name__ = BM0
del BM0
#BLACKMAGIC end import kg.generators (as kg.generators)
#BLACKMAGIC start import formatter (as formatter)
BM3, __name__ = __name__, "BM3"
"""Prints the test data in the correct input format."""

#BLACKMAGIC start import kg.formatters (as kg.formatters)
BM4, __name__ = __name__, "BM4"
import functools

#BLACKMAGIC start import kg.utils.streams (as .utils.streams)
BM5, __name__ = __name__, "BM5"
import collections, enum, functools, io

#BLACKMAGIC start import kg.utils.parsers (as .parsers)
BM6, __name__ = __name__, "BM6"
import collections, decimal, re, string

#BLACKMAGIC start import kg.utils.intervals (as .intervals)
BM7, __name__ = __name__, "BM7"
import collections, collections.abc, enum, functools, itertools, operator


# values represent sorting priority
class BType(enum.IntEnum):
    UE = -3  # upper bound, exclusive
    LI = -1  # lower bound, inclusive
    UI = +1  # upper bound, inclusive
    LE = +3  # lower bound, exclusive

B_BRACKS = {
    BType.UE: ')',
    BType.LI: '[',
    BType.UI: ']',
    BType.LE: '(',
}
BRACK_BTYPE = {bracket: btype for btype, bracket in B_BRACKS.items()}

B_FLIPS = {
    BType.UE: BType.LI,
    BType.LI: BType.UE,
    BType.UI: BType.LE,
    BType.LE: BType.UI,
}

B_NEGS = {
    BType.UE: BType.LE,
    BType.LE: BType.UE,
    BType.UI: BType.LI,
    BType.LI: BType.UI,
}

LO_BTYPES = {BType.LE, BType.LI}
UP_BTYPES = {BType.UE, BType.UI}
LO_BOUND = (-float('inf'), BType.LI) # [-inf
UP_BOUND = (+float('inf'), BType.UI) # +inf]

@functools.lru_cache(maxsize=500)
def _intersect_intervals(a, b):
    def ibounds(a, b):
        ia = ib = 0
        while ia < len(a) and ib < len(b):
            lo = max(a[ia], b[ib])
            up = min(a[ia + 1], b[ib + 1])
            if lo < up:
                yield lo
                yield up
            if up == a[ia + 1]: ia += 2
            if up == b[ib + 1]: ib += 2

    return Intervals(ibounds(a._bds, b._bds))


class Intervals(collections.abc.Hashable): 

    __slots__ = '_bds', '_hash', '_complement'

    def __init__(self, bounds, *, _complement=None):
        self._bds = []
        for bound in bounds:
            if self._bds and self._bds[-1] >= bound:
                raise ValueError("The bounds must be sorted")
            if len(self._bds) % 2 == 0:
                if not Intervals.is_lower(*bound): raise ValueError("At least one of the lower bounds is invalid")
                self._bds.append(bound)
            else:
                if not Intervals.is_upper(*bound): raise ValueError("At least one of the upper bounds is invalid")

                if len(self._bds) > 2 and Intervals.adjacent(*self._bds[-2], *self._bds[-1]):
                    self._bds[-2:] = [bound]
                else:
                    self._bds.append(bound)

        if len(self._bds) % 2:
            raise ValueError("There must be an even number of arguments")
        if self._bds and not (LO_BOUND <= self._bds[0] and self._bds[-1] <= UP_BOUND):
            raise ValueError("The intervals must be a subset of [-inf, +inf]")
        self._hash = None
        self._complement = _complement
        super().__init__()

    def __hash__(self):
        if self._hash is None:
            self._bds = tuple(self._bds)
            self._hash = hash(self._bds) ^ (0xC0FFEE << 3)
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other) and self._bds == other._bds

    def __ne__(self, other):
        return not (self == other)

    def __contains__(self, value):

        if not self._bds:
            return False
        if not Intervals.satisfies(value, *self._bds[0]):
            return False
        if not Intervals.satisfies(value, *self._bds[-1]):
            return False

        l, r = 0, len(self._bds)
        while r - l > 2:
            m = l + r >> 2 << 1 # must be even
            if Intervals.satisfies(value, *self._bds[m]):
                l = m
            elif Intervals.satisfies(value, *self._bds[m - 1]):
                r = m
            else:
                return False

        return True

    def __and__(self, other):
        if hash(self) > hash(other): return other & self
        return _intersect_intervals(self, other)

    def __or__(self, other):
        return ~(~self & ~other)

    def __xor__(self, other):
        return (~self & other) | (self & ~other)

    def __invert__(self):
        if self._complement is None:
            def cbounds(b):
                loi = bool(b and b[0]  == LO_BOUND)
                upi = bool(b and b[-1] == UP_BOUND)
                if not loi: yield LO_BOUND
                for i in range(loi, len(b) - upi):
                    yield Intervals.flip(*b[i])
                if not upi: yield UP_BOUND
            self._complement = Intervals(cbounds(self._bds), _complement=self)
        return self._complement

    @staticmethod
    def is_lower(bound, btype): return btype in LO_BTYPES

    @staticmethod
    def is_upper(bound, btype): return btype in UP_BTYPES

    @staticmethod
    def satisfies(value, bound, btype):
        if btype == BType.UE: return value < bound
        if btype == BType.LI: return bound <= value
        if btype == BType.UI: return value <= bound
        if btype == BType.LE: return bound < value
        assert False #

    @staticmethod
    def adjacent(bound1, btype1, bound2, btype2):
        return bound1 == bound2 and btype2.value - btype1.value == 2

    @staticmethod
    def flip(bound, btype):
        return bound, B_FLIPS[btype]

    def _pieces(self):
        return ((self._bds[i], self._bds[i + 1]) for i in range(0, len(self._bds), 2))

    def __abs__(self):
        return (self & B_NONNEG_INTERVAL) | (-self & B_NONPOS_INTERVAL)

    def __neg__(self):
        return Intervals((-bound, B_NEGS[btype]) for bound, btype in reversed(self._bds))

    def __bool__(self):
        return bool(self._bds)

    def __str__(self):
        return " | ".join(
            f"{B_BRACKS[ltyp]}{lbound}, {rbound}{B_BRACKS[rtyp]}"
            for (lbound, ltyp), (rbound, rtyp) in self._pieces()
        ) if self else "<empty set>"

    def __repr__(self):
        return f"Intervals: {self}"

    @classmethod
    def from_tokens(cls, *tokens):
        if len(tokens) % 4 != 0: raise ValueError("The number of tokens must be a multiple of 4")
        def bounds():
            for i in range(0, len(tokens), 4):
                lch, lvl, uvl, uch = tokens[i:i+4]
                yield lvl, BRACK_BTYPE[lch]
                yield uvl, BRACK_BTYPE[uch]
        return cls(bounds())

    @property
    def lower_bound(self): return self._bds[0][0]  if self._bds else +float('inf')

    @property
    def upper_bound(self): return self._bds[-1][0] if self._bds else -float('inf')



B_FULL_INTERVAL = Intervals([LO_BOUND, UP_BOUND])
B_NONNEG_INTERVAL = Intervals([(0, BType.LI), UP_BOUND])
B_NONPOS_INTERVAL = Intervals([LO_BOUND, (0, BType.UI)])




class VarMeta(type):
    def __pos__(self): return self()
    def __abs__(self): return abs(self())
    def __neg__(self): return -self()
    def __invert__(self): return ~self()

class Var(metaclass=VarMeta):

    __slots__ = 'intervals', '_bd_ct', '_app_pref', '_app'

    def __init__(self, intervals=B_FULL_INTERVAL, *, _bd_ct=0, _app_pref='', _app=()):
        if not isinstance(intervals, Intervals):
            raise TypeError("The first argument must be an Intervals instance")
        self.intervals = intervals
        self._bd_ct = _bd_ct
        self._app_pref = _app_pref
        self._app = tuple(_app)
        super().__init__()

    def _add_bound(self):
        if self._bd_ct >= 2:
            raise RuntimeError("Cannot bound this Var anymore")
        self._bd_ct += 1

    def _add(self, intervals):
        for app in self._app: intervals = app(intervals)
        self.intervals &= intervals
        self._add_bound()
        return self

    def __le__(self, v): return self._add(Intervals([LO_BOUND, (v, BType.UI)]))
    def __lt__(self, v): return self._add(Intervals([LO_BOUND, (v, BType.UE)]))
    def __ge__(self, v): return self._add(Intervals([(v, BType.LI), UP_BOUND]))
    def __gt__(self, v): return self._add(Intervals([(v, BType.LE), UP_BOUND]))
    def __eq__(self, v): return self._add(Intervals([(v, BType.LI), (v, BType.UI)]))
    def __ne__(self, v): return self._add(~Intervals([(v, BType.LI), (v, BType.UI)]))

    def __pos__(self):
        if self._bd_ct: raise TypeError("Cannot get pos if already bounded")
        return self

    def __abs__(self):
        if self._bd_ct: raise TypeError("Cannot get abs if already bounded")
        return Var(self.intervals,
            _app_pref=f"abs {self._app_pref}",
            _app=(Intervals.__abs__, *self._app),
        )

    def __neg__(self):
        if self._bd_ct: raise TypeError("Cannot get neg if already bounded")
        return Var(self.intervals,
            _app_pref=f"neg {self._app_pref}",
            _app=(Intervals.__neg__, *self._app),
        )

    def __invert__(self): return Var(~self.intervals, _bd_ct=2)

    def _combin(self, op, other):
        if isinstance(other, Var): other = other.intervals
        if not isinstance(other, Intervals): return NotImplemented
        return Var(op(self.intervals, other), _bd_ct=2)

    def __and__(self, other): return self._combin(operator.and_, other)
    def __or__ (self, other): return self._combin(operator.or_,  other)
    def __xor__(self, other): return self._combin(operator.xor,  other)

    __rand__ = __and__
    __ror__  = __or__
    __rxor__ = __xor__

    def __str__(self):
        _app_pref = f'{self._app_pref}: ' if self._app_pref else ''
        return f"<{_app_pref}{self.intervals}>"

    __repr__ = __str__



def interval(l, r): return l <= +Var <= r
Interval = interval = warn_on_call("'interval' deprecated; use a <= +Var <= b instead")(interval)

class Bounds(collections.abc.Mapping):
    def __init__(self, bounds=None, **kwbounds):
        if isinstance(bounds, Bounds):
            bounds = bounds._attrs
        self._attrs = {}
        self.accessed = set() # keep track of which attrs were accessed, for validation info purposes.
        for name, value in itertools.chain((bounds or {}).items(), kwbounds.items()):
            if name.startswith('_'):
                raise ValueError("Variable names passed to Bounds cannot start with an underscore")
            if isinstance(value, Var): value = value.intervals  # freeze the Var
            if name in self._attrs:
                raise ValueError("Duplicate names for Bounds not allowed; use '&' instead to combine bounds")
            self._attrs[name] = value
        super().__init__()

    def __and__(self, other): 
        combined = {}
        for attr in sorted(set(self._attrs) | set(other._attrs)):
            def combine(a, b):
                if a is None: return b
                if b is None: return a
                if isinstance(a, Intervals) and isinstance(b, Intervals): return a & b
                if not isinstance(a, Intervals) and not isinstance(b, Intervals): return b
                raise TypeError(f"Conflict for attribute {attr} in merging! {type(a)} vs {type(b)}")
            combined[attr] = combine(self._attrs.get(attr), other._attrs.get(attr))
        return Bounds(combined)

    def __len__(self):  return len(self._attrs)
    def __iter__(self): return iter(self._attrs)

    def __getitem__(self, name):
        if name not in self._attrs: raise KeyError(f"{name} not among the Bounds: {overflow_ell(', '.join(self._attrs))}")
        return getattr(self, name)

    def __getattr__(self, name):
        if name in self._attrs:
            self.accessed.add(name)
            value = self._attrs[name]
            setattr(self, name, value)
            return value
        raise AttributeError

    def __repr__(self): return f'{self.__class__.__name__}({self._attrs!r})'

    def __str__(self): return '{{Bounds:\n{}}}'.format(''.join(f'\t{attr}: {val}\n' for attr, val in self._attrs.items()))





__name__ = BM7
del BM7
#BLACKMAGIC end import kg.utils.intervals (as .intervals)

class ParsingError(Exception): ...

def strict_check_range(x, *args, type="Number"):
    if len(args) == 2:
        l, r = args
        if not (l <= x <= r):
            raise ParsingError(f"{type} {x} not in [{l}, {r}]")
    elif len(args) == 1:
        r, = args
        if isinstance(r, Intervals):
            if x not in r:
                raise ParsingError(f"{type} {x} not in {r}")
        else:
            if not (0 <= x < r):
                raise ParsingError(f"{type} {x} not in [0, {r})")
    elif len(args) == 0:
        pass
    else:
        raise ParsingError(f"Invalid arguments for range check: {args}")
    return x


_int_re = re.compile(r'^(?:0|-?[1-9]\d*)\Z')
intchars = {'-', *string.digits}
def strict_int(x, *args, as_str=False, validate=True): 
    
    # validate
    if validate:
        if not _int_re.fullmatch(x):
            raise ParsingError(f"Expected integer literal, got {x!r}")
    
    # allow to return as string
    if [*args] == ['str']:
        as_str = True
        args = []
    if as_str:
        if args: raise ParsingError("Additional arguments not allowed if as_str is True")
        return x
    
    # parse and check range
    try:
        x = int(x)
    except ValueError as ex:
        raise ParsingError(f"Cannot parse {overflow_ell(x)!r} to int") from ex
    strict_check_range(x, *args, type="Integer")
    return x


_real_re = re.compile(r'^(?P<sign>[+-]?)(?P<int>0?|(?:[1-9]\d*))(?:(?P<dot>\.)(?P<frac>\d*))?\Z')
realchars = intchars | {'+', '-', '.'}

_StrictRealData = collections.namedtuple('_StrictRealData', ['sign', 'dot', 'neg_zero', 'dot_lead', 'dot_trail', 'places'])
def _strict_real_data(x):
    match = _real_re.fullmatch(x)
    if match is None: return None

    sign, int_, dot, frac = map(match.group, ('sign', 'int', 'dot', 'frac'))

    # must have at least one digit
    if not (int_ or frac): return None

    return _StrictRealData(sign=sign, dot=dot,
        neg_zero=sign == '-' and not int_.strip('0') and not frac.strip('0'),
        dot_lead=dot and not int_,
        dot_trail=dot and not frac,
        places=len(frac) if frac else 0,
    )


def strict_real(x, *args, as_str=False, max_places=None, places=None, require_dot=False, allow_plus=False,
        allow_neg_zero=False, allow_dot_lead=False, allow_dot_trail=False, validate=True): 

    # validate
    if validate:
        data = _strict_real_data(x)
        if not data:
            raise ParsingError(f"Expected real literal, got {x!r}")
        if require_dot and not data.dot:
            raise ParsingError(f"Dot required, got {x!r}")
        if not allow_plus and data.sign == '+':
            raise ParsingError(f"Plus sign not allowed, got {x!r}")
        if not allow_neg_zero and data.neg_zero:
            raise ParsingError(f"Real negative zero not allowed, got {x!r}")
        if not allow_dot_lead and data.dot_lead:
            raise ParsingError(f"Real with leading dot not allowed, got {x!r}")
        if not allow_dot_trail and data.dot_trail:
            raise ParsingError(f"Real with trailing dot not allowed, got {x!r}")
        if max_places is not None and data.places > max_places:
            raise ParsingError(f"Decimal place count of {x!r} (={data.places}) exceeds {max_places}")
        if places is not None:
            if isinstance(places, Intervals):
                if data.places not in places:
                    raise ParsingError(f"Decimal place count of {x!r} (={data.places}) not in {places}")
            else:
                if data.places != places:
                    raise ParsingError(f"Decimal place count of {x!r} (={data.places}) not equal to {places}")

    # allow to return as string
    if [*args] == ['str']:
        as_str = True
        args = []
    if as_str:
        if args: raise ParsingError("Additional arguments not allowed if as_str is True")
        return x

    # parse and validate
    try:
        x = decimal.Decimal(x)
    except ValueError as ex:
        raise ParsingError(f"Cannot parse {overflow_ell(x)!r} to Decimal") from ex
    strict_check_range(x, *args, type="Real")
    return x
__name__ = BM6
del BM6
#BLACKMAGIC end import kg.utils.parsers (as .parsers)

class StreamError(Exception): ...
class NoLineError(StreamError): ...
class NoTokenError(StreamError): ...
class NoCharError(StreamError): ...

class ISMode(enum.Enum):
    LINES = 'lines'
    TOKENS = 'tokens'
    RAW_LINES = 'raw_lines'

ISTREAM_DEFAULTS = {
    'extra_chars_allowed': False,
    'ignore_blank_lines': False,
    'ignore_trailing_blank_lines': False,
    'require_trailing_eoln': False,
    'parse_validate': False,
    'line_include_ends': False,
    'line_ignore_trailing_spaces': False,
    'token_skip_spaces': False,
    'token_skip_eolns': False,
    'eoln_skip_spaces': False,
}

ISTREAM_MODE_DEFAULTS = {
    ISMode.LINES: {
        'ignore_trailing_blank_lines': True,
        'line_ignore_trailing_spaces': True,
        'eoln_skip_spaces': True,
    },
    ISMode.TOKENS: {
        'ignore_blank_lines': True,
        'ignore_trailing_blank_lines': True,
        'line_ignore_trailing_spaces': True,
        'token_skip_spaces': True,
        'token_skip_eolns': True,
        'eoln_skip_spaces': True,
    },
    ISMode.RAW_LINES: {
        'line_include_ends': True,
    },
}


class IStreamState:
    def __init__(self, file, *, exc=StreamError):
        self._file = file
        self._exc = exc
        self._buf = ['']
        self._l = 0
        self._i = 0
        self._future1 = None
        self._future2 = None
        super().__init__()

    def next_line(self):
        if self.remaining(): raise RuntimeError("Cannot get buffer next line if not all characters in the current line have been consumed")
        self._l += 1
        self._i = 0
        if self._l == len(self._buf):
            try:
                line = self._file.readline()
            except UnicodeDecodeError as ex:
                raise self._exc("Output stream is not properly encoded") from ex
            else:
                self._buf.append(line)
        if self._future2 is not None and self._future1 != (self._l, self._i):
            self._future1 = None
            self._future2 = None
        if self._future1 is None: self._drop_lines()
        return self._buf[self._l]

    def remaining(self):
        return len(self._buf[self._l]) - self._i

    def peek(self):
        if not self.remaining(): raise RuntimeError("Cannot peek buffer if all characters in the current line have been consumed")
        return self._buf[self._l][self._i]

    def advance(self):
        if not self.remaining(): raise RuntimeError("Cannot advance buffer if all characters in the current line have been consumed")
        self._i += 1

    def consume_line(self):
        buf = self._buf[self._l]
        line = buf[self._i:]
        self._i = len(buf)
        return line

    def consume_until(self, ends):
        i = self._i
        buf = self._buf[self._l]
        while self._i < len(buf) and buf[self._i] not in ends: self._i += 1
        return buf[i:self._i]

    _DROP = 64
    def _drop_lines(self):
        if self._DROP < len(self._buf) < 2 * self._l:
            self._buf = self._buf[self._l:]
            self._l = 0

    def future_begin(self):
        self._drop_lines()
        self._future1 = (self._l, self._i)
        self._future2 = None

    def future_cancel(self):
        (self._l, self._i) = self._future1
        self._future1 = None
        self._future2 = None
        self._drop_lines()

    def future_freeze(self):
        if not (self._future1 and not self._future2):
            raise RuntimeError("Cannot freeze future if it has not yet begun")
        self._future2 = (self._l, self._i)
        (self._l, self._i) = self._future1

    def future_commit(self):
        if not (self._future1 and self._future2):
            raise RuntimeError("Cannot commit future if the future state hasn't been frozen")

        if (self._l, self._i) != self._future1:
            raise RuntimeError("Cannot commit future if the state has changed since the last freeze")

        (self._l, self._i) = self._future2
        self._future1 = None
        self._future2 = None
        self._drop_lines()

class InteractiveStream:
    def __init__(self, reader, writer=None, *, mode=None, exc=StreamError, **options):
        if reader and not reader.readable(): raise OSError('"reader" argument must be writable')
        if writer and not writer.writable(): raise OSError('"writer" argument must be writable')
        if mode is not None and not isinstance(mode, ISMode): raise ValueError(f"Invalid InteractiveStream mode: {mode}")

        self._reader = reader
        self.writer = writer
        self._mode = mode
        self.exc = exc

        self._opts = {**ISTREAM_DEFAULTS}

        if self._mode is not None:
            self._opts.update(ISTREAM_MODE_DEFAULTS[self._mode])

        self._opts.update(options)

        self._token_ends = {SPACE, EOLN}
        self._closed = False
        self._pending = None

        self._buf = IStreamState(self._reader, exc=exc) if self._reader else None
        self._read = ChainRead(self)

        super().__init__()

    @property
    def reader(self): return self._reader

    def _check_open(self):
        if self._closed: raise RuntimeError("InteractiveStream is closed")

    def __iter__(self): return self

    def __next__(self):
        self._check_open()

        if self._pending is not None:
            res = self._pending
            self._pending = None
            self._buf.future_commit()
            return res

        if self._mode == ISMode.TOKENS:
            try:
                return self.read_token(exc=NoTokenError)
            except NoTokenError as ex:
                raise StopIteration from ex
        else:
            try:
                return self.read_line(exc=NoLineError)
            except NoLineError as ex:
                raise StopIteration from ex

    def has_next(self):
        self._check_open()
        try:
            self.peek()
        except StopIteration:
            return False
        else:
            return True

    def peek(self):
        self._check_open()
        if self._pending is None:
            self._buf.future_begin()
            try:
                self._pending = next(self)
            except:
                assert self._pending is None
                self._buf.future_cancel()
                raise
            else:
                assert self._pending is not None
                self._buf.future_freeze()
        return self._pending

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._check_open()
        self.close(exc_type=exc_type)

    def close(self, *, exc_type=None):
        if self._closed: return
        self._pending = None

        try:
            if self.reader and exc_type is None and not self._opts['extra_chars_allowed']:
                if self._opts['ignore_blank_lines'] or self._opts['ignore_trailing_blank_lines']:
                    while True:
                        if self._buf.remaining():
                            try:
                                self._read_eoln_or_eof(exc=NoCharError)
                            except NoCharError as ex:
                                raise self.exc("Extra nonempty lines found at the end") from ex
                        if not self._buffer_line():
                            break
                elif self._buf.remaining() or self._buffer_line():
                    raise self.exc("Extra characters found at the end")
        finally:
            try:
                if self.writer: self.writer.close()
            except BrokenPipeError: # silently allow broken pipe errors
                pass
            finally:
                self._closed = True # can only set this after closing

    def _buffer_line(self):
        buf = self._buf.next_line()
        if self._opts['require_trailing_eoln'] and buf and not buf.endswith(EOLN):
            raise self.exc(f"trailing {stream_char_label(EOLN)} not found")
        return buf

    def read_line(self, *, include_ends=None, ignore_trailing_spaces=None, exc=None):
        self._check_open()
        self._pending = None

        if include_ends is None:
            include_ends = self._opts['line_include_ends']
        if ignore_trailing_spaces is None:
            ignore_trailing_spaces = self._opts['line_ignore_trailing_spaces']
        if include_ends and ignore_trailing_spaces:
            raise ValueError("Cannot ignore trailing spaces if include_ends is true")

        while True:
            if not self._buf.remaining() and not self._buffer_line(): raise (exc or self.exc)("no line found")

            line = self._buf.consume_line()
            assert line

            if line == EOLN and self._opts['ignore_blank_lines']: continue

            # remove undesired trailing whitespace
            if not include_ends:
                last = len(line)
                if last and line[last - 1] == EOLN: last -= 1
                if ignore_trailing_spaces:
                    while last and line[last - 1] == SPACE: last -= 1
                line = line[:last]

            return line

    def read_token(self, *, l=None, ends=None, skip_spaces=None, skip_eolns=None, exc=None):
        self._check_open()
        self._pending = None

        if ends is None:
            ends = self._token_ends
        if skip_spaces is None:
            skip_spaces = self._opts['token_skip_spaces']
        if skip_eolns is None:
            skip_eolns = self._opts['token_skip_eolns']

        ends = force_to_set(ends)

        if not self._buf.remaining(): self._buffer_line()

        while self._buf.remaining() and (
            skip_spaces and self._buf.peek() == SPACE or
            skip_eolns  and self._buf.peek() == EOLN):
            self._buf.advance()
            if not self._buf.remaining(): self._buffer_line()

        if not self._buf.remaining(): raise (exc or self.exc)("no token found")

        res = self._buf.consume_until(ends)
        if l is not None and len(res) not in l: raise self.exc(f"token too long! length must be in {l}")
        return res


    def read_spaces(self):
        self._check_open()
        self._pending = None
        
        while self._buf.remaining() and self._buf.peek() == SPACE: self._buf.advance()


    def read_char(self, target, *, skip_spaces=False, exc=None):
        self._check_open()
        self._pending = None

        if skip_spaces: self.read_spaces()

        if isinstance(target, str):
            if len(target) > 1: raise ValueError(f"Invalid argument for read_char: {target!r}")
            target = {target}
            ret = False
        else:
            target = force_to_set(target)
            ret = True

        if not self._buf.remaining(): self._buffer_line()
        ch = self._buf.peek() if self._buf.remaining() else EOF
        if ch not in target:
            raise (exc or self.exc)(f"{{{', '.join(map(stream_char_label, target))}}} expected but {stream_char_label(ch)} found")

        if ch != EOF: self._buf.advance()

        if ret: return ch

    def _read_eoln_or_eof(self, exc=None):
        return self.read_char({EOLN, EOF}, skip_spaces=self._opts['eoln_skip_spaces'], exc=exc)

    def read_eoln(self, *, skip_spaces=None, exc=None):
        if skip_spaces is None: skip_spaces = self._opts['eoln_skip_spaces']
        return self.read_char(EOLN, skip_spaces=skip_spaces, exc=exc)

    def read_eof(self, *, skip_spaces=None, exc=None):
        if skip_spaces is None: skip_spaces = self._opts['eoln_skip_spaces']
        return self.read_char(EOF, skip_spaces=skip_spaces, exc=exc)

    def read_space(self, exc=None):
        self.read_char(SPACE, exc=exc)


    @listify
    def _do_multiple(self, f, count, *a, cexc=None, **kw):
        if count < 0: raise ValueError(f"n must be nonnegative; got {count}")
        sep = kw.pop('sep', [SPACE])
        end = kw.pop('end', [])
        for i in range(count):
            yield f(*a, **kw)
            if i < count - 1:
                for ch in sep: self.read_char(ch, exc=cexc)
        for ch in end: self.read_char(ch, exc=cexc)

    def read_ints(self, *a, **kw): return self._do_multiple(self.read_int, *a, **kw)
    def read_tokens(self, *a, **kw): return self._do_multiple(self.read_token, *a, **kw)
    def read_reals(self, *a, **kw): return self._do_multiple(self.read_real, *a, **kw)


    def read_int(self, *args, validate=None, **kwargs):
        if validate is None: validate = self._opts['parse_validate']

        int_kwargs = {kw: kwargs.pop(kw) for kw in ('as_str',) if kw in kwargs}
        try:
            return strict_int(self.read_token(**kwargs), *args, validate=validate, **int_kwargs)
        except ParsingError as ex:
            raise self.exc(f"Cannot parse token to int: {', '.join(ex.args)}") from ex


    def read_real(self, *args, validate=None, **kwargs):
        if validate is None: validate = self._opts['parse_validate']

        real_kwargs = {kw: kwargs.pop(kw) for kw in (
            'as_str', 'max_places', 'places', 'require_dot', 'allow_plus', 'allow_neg_zero',
            'allow_dot_lead', 'allow_dot_trail',
        ) if kw in kwargs}
        try:
            return strict_real(self.read_token(**kwargs), *args, validate=validate, **real_kwargs)
        except ParsingError as ex:
            raise self.exc(f"Cannot parse token to real: {', '.join(ex.args)}") from ex


    def __getattr__(self, name):
        if not name.startswith('read_'):
            raise AttributeError(name)
        for tail in ['_eoln', '_eof', '_space', '_spaces']:
            if name.endswith(tail):
                head = name[:-len(tail)] # TODO removesuffix
                break
        else:
            raise AttributeError(name)
        def _meth(self, *a, **kw):
            res = getattr(self, head)(*a, **kw)
            getattr(self, 'read' + tail)()
            return res
        _meth.__name__ = name
        setattr(self.__class__, name, _meth)
        return _meth

    @property
    def read(self): return self._read



    def write(self, *args): return self.writer.write(*args)

    def readable(self, *args):
        return False

    def writable(self, *args): return self.writer.writable(*args)
    def flush(self, *args): return self.writer.flush(*args)
    def isatty(self): return self._reader.isatty() or self.writer.isatty()

    @property
    def closed(self): return self._closed

    @property
    def encoding(self): return self.writer.encoding

    @property
    def errors(self): return self.writer.errors

    @property
    def newlines(self): return self.writer.newlines

    def print(self, *args, **kwargs):
        kwargs.setdefault('file', self.writer)
        return print(*args, **kwargs)




class TextIOPair(io.TextIOBase): 

    def __init__(self, reader, writer):
        if not reader.readable(): raise OSError('"reader" argument must be readable')
        if not writer.writable(): raise OSError('"writer" argument must be writable')
        self.reader = reader
        self.writer = writer
        super().__init__()

    def read(self, *args): return self.reader.read(*args)
    def readline(self, *args): return self.reader.readline(*args)
    def write(self, *args): return self.writer.write(*args)
    def peek(self, *args): return self.reader.peek(*args)
    def readable(self, *args): return self.reader.readable(*args)
    def writable(self, *args): return self.writer.writable(*args)
    def flush(self, *args): return self.writer.flush(*args)

    def close(self):
        try:
            self.writer.close()
        finally:
            self.reader.close()

    def isatty(self): return self.reader.isatty() or self.writer.isatty()

    @property
    def closed(self): return self.writer.closed
    @property
    def encoding(self): return self.writer.encoding
    @property
    def errors(self): return self.writer.errors
    @property
    def newlines(self): return self.writer.newlines

    def print(self, *args, **kwargs):
        kwargs.setdefault('file', self.writer)
        kwargs.setdefault('flush', True)
        return print(*args, **kwargs)

    def input(self): return self.readline().removesuffix('\n')
    
    def __iter__(self): return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line





__name__ = BM5
del BM5
#BLACKMAGIC end import kg.utils.streams (as .utils.streams)

def formatter(f=None, *, print=print):
    def _d(f):
        @functools.wraps(f)
        def _f(file, case, *args, **kwargs):
            with InteractiveStream(None, file) as stream:
                kwargs.setdefault('print', stream.print)
                return f(stream, case, *args, **kwargs)
        return _f
    return _d(f) if f is not None else _d
__name__ = BM4
del BM4
#BLACKMAGIC end import kg.formatters (as kg.formatters)

@formatter
def format_case(stream, cases, *, print, **kwargs):
    print(f"{cases[0]} {cases[1]}")
__name__ = BM3
del BM3
#BLACKMAGIC end import formatter (as formatter)

@listify
def gen_random(rand, *args):
    ... # write your generator here

    # example:
    x, y = map(int, args[:2])

    return rand.randint(-x, x), rand.randint(-y, y)


if __name__ == '__main__':
    write_to_file(format_case, gen_random, argv[1:], stdout)
