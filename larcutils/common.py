import traceback
import logging
from ipaddress import ip_address, ip_interface, ip_network
from typing import Iterable, Hashable, Union
import csv
from pathlib import Path
import builtins
from collections import OrderedDict
import collections.abc

from toolz import curry, pipe, compose
from toolz.curried import (
    map, assoc, dissoc, first as _first, second as _second, last as _last, 
    complement, get as _get, concat, concatv,
)
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from pyrsistent import pmap, pvector, pset, PMap, PVector, PSet

log = logging.getLogger('common')
log.addHandler(logging.NullHandler())

def csv_rows(path: Union[str, Path], header=True):
    with Path(path).expanduser().open() as rfp:
        if header:
            columns = next(csv.reader(rfp))
            for row in csv.DictReader(rfp, columns):
                yield row

@curry
def vcall(func, value):
    'Variadic call'
    return func(*value)

@curry
def vmap(func, seq):
    'Variadic map'
    # return [func(*v) for v in seq]
    return (func(*v) for v in seq)
    # for v in seq:
    #     yield func(*v)

@curry
def vmapcat(func, seq):
    return pipe(concat(func(*v) for v in seq), tuple)

@curry
def mapif(func, seq):
    # return [func(*v) for v in seq]
    if func:
        return (func(v) for v in seq)
    return seq
        
@curry
def vmapif(func, seq):
    # return [func(*v) for v in seq]
    if func:
        return (func(*v) for v in seq)
    return seq


# @curry
# def vmap(func, iterable):
#     return map(lambda v: func(*v), iterable)

def not_null(v):
    return v not in (None, Null())
is_null = complement(not_null)

class Null:
    '''Null pseudo-monad

    Do **not** use as an iterable (i.e. in for loops or over maps), as
    this leads to **infinite loops**.

    '''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Null, cls).__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __contains__(self, *a):
        return False

    def __next__(self):
        return Null()

    def __getattr__(self, key):
        return Null()

    def __getitem__(self, key):
        return Null()

    def __call__(self, *a, **kw):
        return Null()

    def __add__(self, *a):
        return Null()
    def __radd__(self, *a):
        return Null()

    def __sub__(self, *a):
        return Null()
    def __rsub__(self, *a):
        return Null()

    def __mul__(self, *a):
        return Null()
    def __rmul__(self, *a):
        return Null()

    def __div__(self, *a):
        return Null()
    def __rdiv__(self, *a):
        return Null()

def maybe(value, default=None):
    if is_null(value):
        if default is not None:
            return default
        return Null()
    return value

def maybe_pipe(value, *functions, default=None):
    if is_null(value):
        return Null()
    for f in functions:
        try:
            value = f(value)
        except Exception:
            log.error(f'Error in maybe_pipe: \n{traceback.format_exc()}')
            return Null() if default is None else default
        if is_null(value):
            return Null() if default is None else default
    return value

@curry
def short_circuit(function, value):
    if not function(value):
        return Null()
    return value

@curry
def only_if_key(key, func, d):
    if key in d:
        return func(d)
    return d

def first(seq):
    try:
        return _first(seq)
    except StopIteration:
        pass
    return Null()

def second(seq):
    try:
        return _second(seq)
    except StopIteration:
        pass
    return Null()

def last(seq):
    try:
        return _last(seq)
    except StopIteration:
        pass
    return Null()

@curry
def sort_by(func, iterable, **kw):
    return sorted(iterable, key=func, **kw)

@curry
def get(i, indexable, default=None):
    if hasattr(indexable, 'get'):
        return indexable.get(i, default)
    return _get(i, indexable, default)

_getattr = builtins.getattr
@curry
def getattr(attr, obj):
    return _getattr(obj, attr)
    
def do_nothing(value):
    return value

@curry
def update_key(key, value_function, d):
    return assoc(d, key, value_function(d))

@curry
def set_key(key, value, d):
    return assoc(d, key, value)

@curry
def merge_keys(from_: Iterable[Hashable], to: Hashable, value_function, d):
    value = value_function(d)
    return pipe(d, drop_keys(from_), set_key(to, value))

@curry
def drop_key(key: Hashable, d):
    return dissoc(d, key)

@curry
def drop_keys(keys: Iterable[Hashable], d):
    return dissoc(d, *keys)

@curry
def switch_keys(k1, k2, value_function, d):
    return pipe(
        assoc(d, k2, value_function(d)),
        drop_key(k1)
    )

def is_ipv4(ip: (str,int)):
    try:
        return ip_address(ip).version == 4
    except ValueError:
        return False

def is_ip(ip: (str,int)):
    try:
        ip_address(ip)
        return True
    except ValueError:
        return False

def is_interface(iface):
    try:
        ip_interface(iface)
        return True
    except ValueError:
        return False

def is_network(inet):
    try:
        ip_network(inet)
        return True
    except ValueError:
        return False

def ip_to_seq(ip):
    if is_ip(ip):
        return [ip]
    elif is_network(ip):
        return pipe(ip_network(ip).hosts(), map(str), tuple)
    elif is_interface(ip):
        return pipe(ip_interface(ip).network.hosts(), map(str), tuple)
    else:
        log.error(f'Unknown/unparsable ip value: {ip}')
        return []

def sortips(ips):
    return sort_by(ip_address, ips)

def is_dict(d):
    if isinstance(d, collections.abc.Mapping):
        return True
    return False

def is_seq(s):
    if isinstance(s, collections.abc.Iterable):
        return True
    return False
    
def to_pyrsistent(obj):
    if is_dict(obj):
        return pipe(
            obj.items(),
            vmap(lambda k, v: (k, to_pyrsistent(v))),
            pmap,
        )
    elif is_seq(obj):
        return pipe(obj, map(to_pyrsistent), pvector)
    else:
        return obj

def no_pyrsistent(obj):
    if is_dict(obj):
        return pipe(
            obj.items(),
            vmap(lambda k, v: (k, no_pyrsistent(v))),
            dict,
        )
    elif is_seq(obj):
        return pipe(obj, map(no_pyrsistent), tuple)
    else:
        return obj

def flatdict(obj, keys=()):
    if is_dict(obj):
        for k, v in obj.items():
            yield from flatdict(v, keys + (k, ))
    else:
        yield keys + (obj,)
