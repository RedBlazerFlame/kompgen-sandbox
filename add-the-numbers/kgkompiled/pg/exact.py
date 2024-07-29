#!/usr/bin/env python3
from itertools import zip_longest
#BLACKMAGIC start import kg.checkers (as kg.checkers)
BM0, __name__ = __name__, "BM0"
import argparse, contextlib, functools, os, os.path, sys, traceback

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
#BLACKMAGIC start import kg.utils.streams (as .utils.streams)
BM3, __name__ = __name__, "BM3"
import collections, enum, functools, io

#BLACKMAGIC start import kg.utils.parsers (as .parsers)
BM4, __name__ = __name__, "BM4"
import collections, decimal, re, string

#BLACKMAGIC start import kg.utils.intervals (as .intervals)
BM5, __name__ = __name__, "BM5"
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





__name__ = BM5
del BM5
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
__name__ = BM4
del BM4
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





__name__ = BM3
del BM3
#BLACKMAGIC end import kg.utils.streams (as .utils.streams)
#BLACKMAGIC start import kg.utils.judging (as .utils.judging)
BM6, __name__ = __name__, "BM6"
import functools, json


class ParseError(Exception): ...
class Wrong(Exception): ...
class Fail(Exception): ...

WA = Wrong

class Verdict:
    AC = "Success"
    
    PAE = "Wrong answer (Parse error)"
    
    CE = "Compile Error"
    
    WA = "Wrong answer"
    
    RTE = "Runtime Error"
    
    TLE = "Time Limit Exceeded"
    
    EXC = "Checker/Interactor raised an error [BAD!]"
    
    FAIL = "Checker/Interactor failed [BAD!]"


polygon_rcode = {
    Verdict.AC: 0,
    Verdict.CE: 1,
    Verdict.PAE: 2,
    Verdict.WA: 1,
    Verdict.RTE: 1,
    Verdict.TLE: 1,
    Verdict.FAIL: 3,
    Verdict.EXC: 3,
}
polygon_partial = 16

kg_rcode = {
    Verdict.AC: 0,
    Verdict.CE: 11,
    Verdict.WA: 21,
    Verdict.RTE: 22,
    Verdict.TLE: 23,
    Verdict.PAE: 24,
    Verdict.FAIL: 31,
    Verdict.EXC: 32,
}




xml_outcome = {
    Verdict.AC: "Accepted",
    Verdict.CE: "No - Compilation Error",
    Verdict.PAE: "No - Wrong Answer",
    Verdict.WA: "No - Wrong Answer",
    Verdict.RTE: "No - Run-time Error",
    Verdict.TLE: "No - Time Limit Exceeded",
    Verdict.FAIL: "No - Other - Contact Staff",
    Verdict.EXC: "No - Other - Contact Staff",
}

hr_verdict_name = {
    Verdict.AC: "Success",
    Verdict.CE: "Compilation Error",
    Verdict.PAE: "Wrong Answer",
    Verdict.WA: "Wrong Answer",
    Verdict.RTE: "Runtime Error",
    Verdict.TLE: "Time limit exceeded", # I don't like HR's message "terminated due to timeout"
    Verdict.FAIL: "Checker Failed",
    Verdict.EXC: "Checker Failed.", # Added a dot so we can recognize which kind of failure it is.
}


def write_json_verdict(verdict, message, score, result_file):
    with open(result_file, 'w') as f:
        json.dump({'verdict': verdict, 'message': message, 'score': float(score)}, f)

def write_xml_verdict(verdict, message, score, result_file):
    from xml.etree.ElementTree import Element, ElementTree
    result = Element('result')
    result.set('security', result_file)
    result.set('outcome', xml_outcome[verdict])
    result.text = str(verdict) + ": " + message
    ElementTree(result).write(result_file, xml_declaration=True, encoding="utf-8")


def minimum_score(scores, mn=0.0, mx=1.0, break_on_min=False, exc=Fail):
    if mn > mx: raise exc(f"Invalid arguments for mn and mx: {mn} > {mx}")
    m = mx
    to_exit = lambda: break_on_min and m == mn
    if not to_exit():
        for score in scores:
            if score is None:
                raise exc("A score of 'None' was returned.")
            if not (mn <= score <= mx):
                raise exc(f"Invalid score: {score}. It must be in the interval [{mn}, {mx}].")
            m = min(m, score)
            if to_exit(): break # we can stop now
    return m


def average_score(scores, exc=Fail):
    tl = ct = 0
    for score in scores:
        if score is None: raise exc("A score of 'None' was returned.")
        tl += score
        ct += 1
    if ct == 0: raise exc("Cannot take average of empty score list")
    return tl / ct



def on_exhaust(exc):
    def _d(f):
        @functools.wraps(f)
        def _f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except StopIteration as st:
                raise exc from st
        return _f
    return _d



__name__ = BM6
del BM6
#BLACKMAGIC end import kg.utils.judging (as .utils.judging)

class CheckerError(Exception): ...

class Checker:
    def __call__(self, input_file, output_file, judge_file, *args, **kwargs):
        with contextlib.ExitStack() as stack:
            input_s = stack.enter_context(InteractiveStream(
                input_file,
                mode=ISMode(self.input_mode),
                exc=lambda message: Fail(f'[input] {message}'),
                **self.stream_settings['input'],
            ))
            output_s = stack.enter_context(InteractiveStream(
                output_file,
                mode=ISMode(self.output_mode),
                exc=ParseError,
                **self.stream_settings['output'],
            ))
            judge_s = stack.enter_context(InteractiveStream(
                judge_file,
                mode=ISMode(self.judge_mode),
                exc=lambda message: Fail(f'[judge] {message}'),
                **self.stream_settings['judge'],
            )) if judge_file else None

            return self.check(input_s, output_s, judge_s, *args, **kwargs)

    def init(self, *args, **kwargs):

        if not args: args = ['lines']
        if len(args) == 1: args = args*3
        if len(args) != 3: raise ValueError(f"Invalid args: {args}")
        self.input_mode, self.output_mode, self.judge_mode = args

        valid_fields = {'input', 'output', 'judge'}
        def to_fields(arg):
            value = kwargs.pop(arg, False)
            value = {*value} if not isinstance(value, bool) else valid_fields if value else set()
            if not value <= valid_fields:
                raise ValueError(f"Invalid {arg} argument(s): {value - valid_fields}")
            return value

        if 'no_extra_chars' in kwargs:
            if 'extra_chars_allowed' in kwargs:
                raise ValueError("'no_extra_chars' and 'extra_chars_allowed' not allowed together")
            kwargs['extra_chars_allowed'] = valid_fields - to_fields('no_extra_chars')
        kwargs.setdefault('extra_chars_allowed', {'input', 'judge'})

        fields = [(key, to_fields(key)) for key in [*kwargs]]
        self.stream_settings = {type_: {key: True for key, types in fields if type_ in types} for type_ in valid_fields}

    @classmethod
    def from_func(cls, *args, **kwargs):
        f, args = pop_callable(args)
        def _checker(f):
            check = cls()
            check.check = f
            check.init(*args, **kwargs)
            return check
        return _checker(f) if f is not None else _checker


class BuiltChecker(Checker):
    _names = {'get_one_input', 'get_output_for_input', 'get_judge_data_for_input', 'aggregate', 'iterate', 'check_one', 'wrap_up'}
    _aliases = {
        'get_output_from_input': 'get_output_for_input',
        'get_judge_data_from_input': 'get_judge_data_for_input',
        'iterator': 'iterate',
    }
    def __init__(self):
        for name in self._names: setattr(self, name, None)
        super().__init__()

    def init(self, *args, cases=None, **kwargs):
        if cases is None:
            pass
        elif cases == 'multi':
            self._set('iterate', checker_iterate_with_casecount)
        elif cases == 'single':
            self._set('iterate', checker_iterate_single)
        else:
            raise ValueError(f"Unknown 'cases' argument: {cases}")
        super().init(*args, **kwargs)

    def _set(self, name, arg):
        if name in self._aliases:
            name, wrong_name = self._aliases[name], name
        if name not in self._names:
            raise ValueError(f"Unknown name to set: {name}")
        if getattr(self, name):
            raise ValueError(f"{name} already set!")
        setattr(self, name, arg)
        return arg

    def check(self, *args, **kwargs):
        return CheckingContext(self, *args, **kwargs)()




class CheckingContext:
    def __init__(self, checker, input_s, output_s, judge_s, **kwargs):
        self.checker = checker
        self.input_stream  = input_s
        self.output_stream = output_s
        self.judge_stream  = judge_s
        self.kwargs = kwargs
        self.input_file  = deprec_alias('input_file', self.input_stream, new_name='input_stream')
        self.output_file = deprec_alias('output_file', self.output_stream, new_name='output_stream')
        self.judge_file  = deprec_alias('judge_file', self.judge_stream, new_name='judge_stream')
        super().__init__()

    @on_exhaust(Fail("Input stream fully read but expected more"))
    def get_one_input(self, **kwargs):
        return self.checker.get_one_input(self.input_stream, exc=Fail, **kwargs, **self.kwargs)

    @on_exhaust(ParseError("Contestant output stream fully read but expected more"))
    def get_output_for_input(self, input, **kwargs):
        return self.checker.get_output_for_input(self.output_stream, input, exc=ParseError, **kwargs, **self.kwargs)

    @on_exhaust(Fail("Judge data stream fully read but expected more"))
    def get_judge_data_for_input(self, input, **kwargs):
        get_data = self.checker.get_judge_data_for_input or self.checker.get_output_for_input
        if not get_data: return
        return get_data(self.judge_stream, input, exc=Fail, **kwargs, **self.kwargs)

    @on_exhaust(Fail("aggregate function failed"))
    def aggregate(self, scores):
        return (self.checker.aggregate or minimum_score)(scores)

    @on_exhaust(Fail("iterate function failed"))
    def iterate(self):
        return (self.checker.iterate or checker_iterate_with_casecount)(self)

    @on_exhaust(Fail("check_one function failed"))
    def check_one(self, input, output, judge_data, **kwargs):
        return self.checker.check_one(input, output, judge_data, **kwargs, **self.kwargs)

    # warn on alias usage
    get_score       = deprec_alias('get_score', check_one)
    next_input      = deprec_alias('next_input', get_one_input)
    next_output     = deprec_alias('next_output', get_output_for_input)
    next_judge_data = deprec_alias('next_judge_data', get_judge_data_for_input)

    @on_exhaust(Fail("wrap_up function failed"))
    def wrap_up(self, success, score, raised_exc, **kwargs):
        if not self.checker.wrap_up: return
        return self.checker.wrap_up(success, score=score, raised_exc=raised_exc, **kwargs, **self.kwargs)

    def __call__(self):
        try:
            score = self.aggregate(self.iterate())
        except (Wrong, ParseError) as exc:
            self.wrap_up(success=False, score=None, raised_exc=exc)
            raise
        else:
            new_score = self.wrap_up(success=True, score=score, raised_exc=None)
            return new_score if new_score is not None else score


def make_checker_builder():
    return Builder(name='checker', build_standalone=Checker.from_func, build_from_parts=BuiltChecker)

checker = make_checker_builder()


def checker_iterate_with_casecount(it):
    t = int(next(it.input_stream))
    for cas in range(t):
        inp = it.get_one_input(caseno=cas)
        yield it.check_one(inp,
            it.get_output_for_input(inp, caseno=cas),
            it.get_judge_data_for_input(inp, caseno=cas),
            caseno=cas,
        )


def checker_iterate_single(it, *, cas=0):
    inp = it.get_one_input(caseno=cas)
    yield it.check_one(inp,
        it.get_output_for_input(inp, caseno=cas),
        it.get_judge_data_for_input(inp, caseno=cas),
        caseno=cas,
    )





class OldChecker:
    def __init__(self):
        self.pending = BuiltChecker()
        self.checker = None
        for name in (*self.pending._names, *self.pending._aliases):
            setattr(self, name, functools.partial(self._set, name))
        self.init_args = None
        self.init_kwargs = None
        super().__init__()

    def _set(self, name, arg):
        if self.checker: raise RuntimeError("Cannot change checker anymore once called")
        return self.pending._set(name, arg)

    def set_single_checker(self, *args, **kwargs):
        return self.set_checker(*args, cases='single', **kwargs)

    def set_multi_checker(self, *args, **kwargs):
        return self.set_checker(*args, cases='multi', **kwargs)

    def set_checker(self, *args, **kwargs):
        f, args = pop_callable(args)
        self.init_args = args
        self.init_kwargs = kwargs
        def _set_checker(f):
            self._set('check_one', f)
            return f
        return _set_checker(f) if f is not None else _set_checker

    def __call__(self, *args, **kwargs):
        if self.checker is None:
            self.checker = self.pending
            self.checker.init(*self.init_args, **self.init_kwargs)
            self.pending = None
        return check_files(self.checker, *args, **kwargs)

chk = OldChecker() # create singleton
_warner = lambda name: lambda f: f
set_single_checker = _warner('set_single_checker')(chk.set_single_checker)
set_multi_checker = _warner('set_multi_checker')(chk.set_multi_checker)
@_warner('set_checker')
def set_checker(*args, **kwargs):
    chk.checker = None
    chk.pending = None
    f, args = pop_callable(args)
    def _set_checker(f):
        chk.checker = Checker.from_func(f, *args, **kwargs)
        chk.pending = None
        return chk.checker
    return _set_checker(f) if f is not None else _set_checker








def _check_generic(check, input=None, output=None, judge=None, **kwargs):

    if CURR_PLATFORM in {'cms', 'cms-it'}:
        def handle(exc, verdict):
            return verdict, getattr(exc, 'score', 0.0), ""
    else:
        def handle(exc, verdict):
            if kwargs.get('verbose'): traceback.print_exc(limit=-1) 
            return verdict, getattr(exc, 'score', 0.0), str(exc)

    with contextlib.ExitStack() as stack:

        def maybe_open(arg, mode='r'):
            if arg is None:
                return None, None
            if isinstance(arg, str):
                return arg, stack.enter_context(open(arg, mode))
            return arg

        kwargs['input_path'],  input_f  = maybe_open(input)
        kwargs['output_path'], output_f = maybe_open(output)
        kwargs['judge_path'],  judge_f  = maybe_open(judge)

        try:
            score = check(input_f, output_f, judge_f, **kwargs)
            if not (0.0 <= score <= 1.0):
                raise CheckerError(f"The checker returned an invalid score: {score!r}")
            return Verdict.AC if score > 0 else Verdict.WA, score, ""
        except ParseError as exc:
            return handle(exc, Verdict.PAE)
        except Wrong as exc:
            return handle(exc, Verdict.WA)
        except Fail as exc:
            return handle(exc, Verdict.FAIL)
        except Exception as exc:
            return handle(exc, Verdict.EXC)










_plat_checkers = {}
def _reg_plat_checker(name):
    def reg(f):
        assert name not in _plat_checkers, f"{name} registered twice!"
        _plat_checkers[name] = f
        return f
    return reg






@_reg_plat_checker('local')
@_reg_plat_checker('kg')
@_reg_plat_checker('pg')
@_reg_plat_checker('pc2')
def _check_local(check, *, title='', log_file=sys.stdout, help=None, force_verbose=False, exit_after=True):
    desc = help or CURR_PLATFORM + (' checker for the problem' + (f' "{title}"' if title else ''))
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_path', help='input file path')
    parser.add_argument('output_path', help="contestant's file path")
    parser.add_argument('judge_path', help='judge auxiliary data file path')
    parser.add_argument('result_file', nargs='?', help='target file to contain the verdict in XML format')
    parser.add_argument('extra_args', nargs='*', help='extra arguments that will be ignored')
    parser.add_argument('-C', '--code', default='n/a', help='path to the solution used')
    parser.add_argument('-t', '--tc-id', default=None, type=int, help='test case ID, zero indexed')
    if CURR_PLATFORM == 'pc2':
        parser.add_argument('-q', '--quiet', action='store_true', help='print less details')
    else:
        parser.add_argument('-v', '--verbose', action='store_true', help='print more details')
    parser.add_argument('-i', '--identical', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    verbose = force_verbose or (not args.quiet if CURR_PLATFORM == 'pc2' else args.verbose)
    tc_id = args.tc_id or ''

    if verbose:
        if args.extra_args:
            warn_print(f"{tc_id:>3} [C] Received extra args {args.extra_args}... ignoring them.", file=log_file)
        print(f"{tc_id:>3} [C] Checking the output...", file=log_file)

    verdict, score, message = _check_generic(check,
        input=args.input_path,
        output=args.output_path,
        judge=args.judge_path,
        code_path=args.code,
        tc_id=args.tc_id,
        identical=args.identical,
        verbose=verbose,
    )

    if verbose:
        print(f"{tc_id:>3} [C] Result:  {verdict}", file=log_file)
        print(f"{tc_id:>3} [C] Score:   {score}", file=log_file)
        if message: print(f"{tc_id:>3} [C] Message: {overflow_ell(message, 100)}", file=log_file)
    else:
        print(f"{tc_id:>3} [C] Score={score} {verdict}", file=log_file)

    if args.result_file:
        if verbose: print(f"{tc_id:>3} [C] Writing result to '{args.result_file}'...", file=log_file)
        write_xml_verdict(verdict, message, score, args.result_file)

    if CURR_PLATFORM == 'pc2':
        exit_code = polygon_rcode[verdict]
    elif CURR_PLATFORM == 'pg':
        exit_code = polygon_partial + int(score * 100) if 0 < score < 1 else polygon_rcode[verdict]
    else:
        exit_code = kg_rcode[verdict]

    if exit_after: exit(exit_code)

    return exit_code








# TODO argv thing
def check_files(check, *args, platform=CURR_PLATFORM, **kwargs):
    return _plat_checkers[platform](check, *args, **kwargs)



__name__ = BM0
del BM0
#BLACKMAGIC end import kg.checkers (as kg.checkers)

def is_exactly_equal(seq1, seq2):
    return all(val1 == val2 for val1, val2 in zip_longest(seq1, seq2))

@checker(extra_chars_allowed=['input'])
@default_score
def check_exactly_equal(input_file, output_file, judge_file, **kwargs):
    output_lines = list(output_file)
    judge_lines = list(judge_file)
    if not is_exactly_equal(output_lines, judge_lines):
        raise Wrong('Incorrect.')

if __name__ == '__main__': check_files(check_exactly_equal, help="Exact diff checker")
