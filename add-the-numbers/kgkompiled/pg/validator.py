#!/usr/bin/env python3
"""Checks whether the input file is valid."""

from sys import *
#BLACKMAGIC start import kg.validators (as kg.validators)
BM0, __name__ = __name__, "BM0"
import argparse, functools, io, itertools, re, sys

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
#BLACKMAGIC start import kg.utils.intervals (as .utils.intervals)
BM3, __name__ = __name__, "BM3"
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





__name__ = BM3
del BM3
#BLACKMAGIC end import kg.utils.intervals (as .utils.intervals)
#BLACKMAGIC start import kg.utils.parsers (as .utils.parsers)
BM4, __name__ = __name__, "BM4"
import collections, decimal, re, string


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
__name__ = BM4
del BM4
#BLACKMAGIC end import kg.utils.parsers (as .utils.parsers)

_patterns = functools.lru_cache(maxsize=None)(re.compile)


class ValidationError(Exception): ...
class ValidationStreamError(Exception): ... # TODO unify with streams.StreamError


class StrictInputStream:
    def __init__(self, file, *, interactive=False):
        self.last = None
        self.next = None
        if not interactive and not isinstance(file, io.StringIO):
            file = io.StringIO(file.read())
        self.file = file
        self._read = ChainRead(self)
        super().__init__()

    @classmethod
    def from_string(self, s):
        return StrictInputStream(io.StringIO(s))


    def _next_char(self):
        if self.last == EOF: raise ValidationStreamError("Read past EOF")
        if self.next is None: self.next = self.file.read(1)
        self.last = self.next
        self.next = None
        return self.last

    def peek_char(self):
        if self.last == EOF: raise ValidationStreamError("Peeked past EOF")
        if self.next is None: self.next = self.file.read(1)
        return self.next

    def _read_cond(self, good, bad, *, l=None, n=None, maxn=None, include_end=False, _called="_read_cond"):
        if maxn is None: maxn = (1 << 200) # 'infinite' enough for our purposes
        if l is not None:
            if not isinstance(l, Intervals):
                raise TypeError("Invalid type for l; must be intervals")
            maxn = int(min(maxn, l.upper_bound + 1))
        if maxn < 0:
            raise ValueError(f"maxn must be nonnegative; got {maxn}")
        res = io.StringIO()
        lres = 0
        while good(self.peek_char()):
            if bad(self.peek_char()):
                raise ValidationStreamError(f"Invalid character for {_called} detected: {stream_char_label(self.peek_char())}")
            res.write(self._next_char())
            lres += 1
            if n is not None and lres > n: 
                raise ValidationStreamError(f"Expected exactly {n} characters, got more.")
            if lres > maxn:
                raise ValidationStreamError(f"Took too many characters! Expected at most {maxn}")
        if n is not None and lres != n:
            raise ValidationStreamError(f"Expected exactly {n} characters, got {lres}")
        if l is not None and lres not in l:
            raise ValidationStreamError(f"Expected length in {l}, got {lres}")
        if include_end:
            res.write(self._next_char())
        return res.getvalue()

    def read_until(self, ends, *, other_ends=set(), charset=set(), _called="read_until", **kwargs):
        ends = force_to_set(ends)
        other_ends = force_to_set(other_ends)
        charset = force_to_set(charset)
        return self._read_cond(
            lambda ch: ch not in ends and ch not in other_ends,
            lambda ch: charset and ch not in charset,
            _called=_called,
            **kwargs,
        )

    def read_while(self, charset, *, ends=set(), _called="read_while", **kwargs):
        ends = force_to_set(ends)
        charset = force_to_set(charset)
        return self._read_cond(
            lambda ch: ch in charset,
            lambda ch: ch in ends,
            _called=_called,
            **kwargs,
        )

    def read_line(self, *, eof=False, _called="line", **kwargs):
        return self.read_until({EOLN, EOF} if eof else {EOLN}, _called=_called, **kwargs)

    def read_token(self, regex=None, *, ends={SPACE, EOLN, EOF}, other_ends=set(), _called="token", **kwargs): # optimize this. 
        tok = self.read_until(ends, other_ends=other_ends, _called=_called, **kwargs)
        if regex is not None and not _patterns('^' + regex + r'\Z').fullmatch(tok):
            raise ValidationStreamError(f"Expected token with regex {regex!r}, got {tok!r}")
        return tok

    @listify
    def _do_multiple(self, f, count, *a, **kw):
        if count < 0: raise ValueError(f"n must be nonnegative; got {count}")
        sep = kw.pop('sep', [SPACE])
        end = kw.pop('end', [])
        for i in range(count):
            yield f(*a, **kw)
            if i < count - 1:
                for ch in sep: self.read_char(ch)
        for ch in end: self.read_char(ch)

    def read_ints(self, *a, **kw): return self._do_multiple(self.read_int, *a, **kw)
    def read_tokens(self, *a, **kw): return self._do_multiple(self.read_token, *a, **kw)
    def read_reals(self, *a, **kw): return self._do_multiple(self.read_real, *a, **kw)


    def read_int(self, *args, **kwargs):
        int_kwargs = {kw: kwargs.pop(kw) for kw in ('as_str',) if kw in kwargs}
        return strict_int(self.read_token(charset=intchars, _called="int", **kwargs), *args, **int_kwargs)

    def read_real(self, *args, **kwargs):
        real_kwargs = {kw: kwargs.pop(kw) for kw in (
            'as_str', 'max_places', 'places', 'require_dot', 'allow_plus',
            'allow_neg_zero', 'allow_dot_lead', 'allow_dot_trail',
        ) if kw in kwargs}
        return strict_real(self.read_token(charset=realchars, _called="real", **kwargs), *args, **real_kwargs)

    def read_space(self): return self.read_char(SPACE)
    def read_eoln(self): return self.read_char(EOLN)
    def read_eof(self): return self.read_char(EOF)

    def read_char(self, target):
        if isinstance(target, str):
            if len(target) > 1:
                raise ValueError(f"Invalid argument for read_char: {target!r}")
            if self._next_char() != target:
                raise ValidationStreamError(f"Expected {stream_char_label(target)}, got {stream_char_label(self.last)}")
        else:
            target = force_to_set(target)
            if self._next_char() not in target:
                raise ValidationStreamError(f"Expected [{', '.join(map(stream_char_label, target))}], got {stream_char_label(self.last)}")
            return self.last

    def __getattr__(self, name):
        if not name.startswith('read_'):
            raise AttributeError
        for tail in ['_eoln', '_eof', '_space']:
            if name.endswith(tail):
                head = name[:-len(tail)] # TODO removesuffix
                break
        else:
            raise AttributeError
        def _meth(self, *a, **kw):
            res = getattr(self, head)(*a, **kw)
            getattr(self, 'read' + tail)()
            return res
        _meth.__name__ = name
        setattr(self.__class__, name, _meth)
        return getattr(self, name)

    @property
    def read(self):
        return self._read

StrictStream = StrictInputStream


def validator(f=None, *, bounds=None, subtasks=None, extra_chars_allowed=False, suppress_eof_warning=None):

    def _d(f):
        @functools.wraps(f)
        def _f(file, *args, force_subtask=False, interactive=False, **kwargs):
            if force_subtask and not (subtasks and 'subtask' in kwargs and kwargs['subtask'] in subtasks):
                raise RuntimeError(f"invalid subtask given: {kwargs.get('subtask')!r}")
            stream = StrictInputStream(file, interactive=interactive)
            if bounds is not None or subtasks is not None:
                lim = Bounds(kwargs.get('lim'))
                if bounds: lim &= Bounds(bounds)
                if subtasks: lim &= Bounds(subtasks.get(kwargs['subtask']))
                kwargs['lim'] = lim
            res = f(stream, *args, **kwargs)
            if stream.last != EOF and not extra_chars_allowed:
                stream.read_eof()
            return res
        return _f

    return _d(f) if f is not None else _d

def detect_subtasks(validate, file, subtasks, *args, **kwargs):
    file = io.StringIO(file.read())
    for subtask in subtasks:
        file.seek(0)
        try:
            validate(file, *args, subtask=subtask, force_subtask=True, **kwargs)
        except Exception:
            ... 
        else:
            yield subtask

def validate_or_detect_subtasks(validate, subtasks, file=sys.stdin, outfile=sys.stdout, *args, title='', **kwargs):
    desc = CURR_PLATFORM + ' validator for the problem' + (f' "{title}"' if title else '')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('subtask', nargs='?', help='which subtask to check the file against')
    parser.add_argument('--detect-subtasks', '-d', action='store_true', help='detect subtasks instead')
    pargs, unknown = parser.parse_known_args()
    subtask = pargs.subtask


    if pargs.detect_subtasks:
        print(*detect_subtasks(validate, file, subtasks, *args, **kwargs), file=outfile)
    else:
        validate(file, *args, subtask=subtask, **kwargs)

__name__ = BM0
del BM0
#BLACKMAGIC end import kg.validators (as kg.validators)

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
