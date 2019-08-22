import os
import sys
import io
import traceback
import importlib
import logging
from ipaddress import ip_address, ip_interface, ip_network
from typing import Iterable, Hashable, Union
import csv
from pathlib import Path
import builtins
import collections
import functools
import re
import random
import json
import inspect
import math
import itertools
from datetime import datetime

from multipledispatch import dispatch
import pyperclip
import jmespath
import dateutil.parser
import networkx as nx
from pkg_resources import resource_filename as _resource_filename

try:
    from cytoolz import curry, pipe, compose, merge, concatv
    from cytoolz.curried import (
        map, mapcat, assoc, dissoc, valmap, partial, topk,
        first as _first, second as _second, last as _last,
        complement, get as _get, concat, filter, do, groupby,
        take, memoize,
    )
except ImportError:
    from toolz import curry, pipe, compose, merge, concatv
    from toolz.curried import (
        map, mapcat, assoc, dissoc, valmap, partial, topk,
        first as _first, second as _second, last as _last,
        complement, get as _get, concat, filter, do, groupby,
        take, memoize,
    )

from pyrsistent import pmap, pvector, PVector

resource_filename = curry(_resource_filename)(__name__)
data_path = compose(
    Path,
    resource_filename,
    str,
    lambda p: Path('data', p),
)

log = logging.getLogger('common')
log.addHandler(logging.NullHandler())

anypath = Union[str, Path]

ip_re = re.compile(r'(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])(?![\d\.]+)')
ip_only_re = re.compile(f'^{ip_re.pattern}$')

def mini_tb(levels=3):
    frame = inspect.currentframe().f_back
    parents = [frame.f_back]
    for i in range(levels - 1):
        if parents[-1].f_back:
            parents.append(parents[-1].f_back)
        else:
            break
    return '\n' + pipe(
        parents,
        map(inspect.getframeinfo),
        vmap(lambda filen, lnum, fname, lns, i: (
            f'{Path(filen).name}', lnum, fname, lns[i],
        )),
        vmap(lambda path, lnum, fname, line: (
            f'- {fname} | {path}:{lnum} | {line.rstrip()}'
        )),
        '\n'.join,
    )

@curry
def jmes(search, d):
    if is_null(d):
        log.error(
            f'null dict passed to jmes {mini_tb(5)}'
        )
        return Null
    return jmespath.search(search, d)

def parse_dt(ts: str, local=False):
    return dateutil.parser.parse(
        ts
    ).astimezone(dateutil.tz.tzlocal())

@curry
def max(iterable, **kw):
    return builtins.max(iterable, **kw)

@curry
def min(iterable, **kw):
    return builtins.min(iterable, **kw)

@curry
def csv_rows_from_path(path: anypath, *, header=True, columns=None, **kw):
    return csv_rows_from_fp(Path(path).expanduser().open(), header=header, **kw)
csv_rows = csv_rows_from_path

@curry
def csv_rows_from_content(content, *, header=True, columns=None, **kw):
    return csv_rows_from_fp(io.StringIO(content), header=header, **kw)

@curry
def csv_rows_from_fp(rfp, *, header=True, columns=None, **kw):
    if header:
        columns = next(csv.reader(rfp))
        reader = csv.DictReader(rfp, columns, **kw)
    elif is_seq(columns):
        reader = csv.DictReader(rfp, columns, **kw)
    else:
        reader = csv.reader(rfp, **kw)
    for row in pipe(reader, filter(None)):
        yield row
    
@curry
def write_csv(path: anypath, rows, *, columns=None, **kw):
    test, rows = itertools.tee(rows)
    row = next(test)
    if not is_dict(row) and columns is not None:
        rows = pipe(
            rows,
            map(lambda r: {c: r[i] for i, c in enumerate(columns)}),
        )
    with Path(path).expanduser().open('w') as wfp:
        if columns:
            writer = csv.DictWriter(wfp, columns, **kw)
            writer.writeheader()
        else:
            writer = csv.writer(wfp, **kw)
        writer.writerows(rows)

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
    
starmap = vmap

@curry
def vfilter(func, seq):
    return filter(vcall(func), seq)

@curry
def vmapcat(func, seq):
    return pipe(concat(func(*v) for v in seq), tuple)

@curry
def vgroupby(key, seq):
    return groupby(vcall(key), seq)

@curry
def vvalmap(func, d, factory=dict):
    return valmap(vcall(func), d, factory=factory)

@curry
def select(keys, iterable):
    for indexable in iterable:
        yield tuple(indexable[k] for k in keys)

@curry
def find(find_func, iterable):
    for value in iterable:
        if find_func(value):
            return value
    return Null

@curry
def vfind(find_func, iterable):
    for value in iterable:
        if find_func(*value):
            return value
    return Null

@curry
def index(find_func, iterable):
    for i, value in enumerate(iterable):
        if find_func(value):
            return i
    return Null

@curry
def vindex(find_func, iterable):
    return index(vcall(find_func))

@curry
def callif(if_func, func, value):
    if if_func(value):
        return func(value)

@curry
def vcallif(if_func, func, value):
    if if_func(*value):
        return func(*value)

@curry
def mapdo(func, iterable):
    values = tuple(iterable)
    for v in values:
        func(v)
    return values

@curry
def vmapdo(func, iterable):
    values = tuple(iterable)
    for v in values:
        func(*v)
    return values

@curry
def mapdo(func, iterable):
    for v in iterable:
        func(v)
    return iterable

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

class _null:
    '''Null pseudo-monad

    Do **not** use as an iterable (i.e. in for loops or over maps), as
    this leads to **infinite loops**.

    '''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_null, cls).__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'Null'

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __contains__(self, *a):
        return False

    def __next__(self):
        return Null

    def __getattr__(self, key):
        return Null

    def __getitem__(self, key):
        return Null

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return False

    def __le__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __call__(self, *a, **kw):
        return Null

    def __add__(self, *a):
        return Null

    def __radd__(self, *a):
        return Null

    def __sub__(self, *a):
        return Null

    def __rsub__(self, *a):
        return Null

    def __mul__(self, *a):
        return Null

    def __rmul__(self, *a):
        return Null

    def __div__(self, *a):
        return Null

    def __rdiv__(self, *a):
        return Null

Null = _null()

def not_null(v):
    return v not in (None, Null)
is_null = complement(not_null)

def maybe(value, default=None):
    if is_null(value):
        if default is not None:
            return default
        return Null
    return value

def maybe_pipe(value, *functions, default=None):
    if is_null(value):
        return Null
    for f in functions:
        try:
            value = f(value)
        except Exception:
            log.error(f'Error in maybe_pipe: \n{traceback.format_exc()}')
            return Null if default is None else default
        if is_null(value):
            return Null if default is None else default
    return value

def maybe_int(value, default=None):
    if is_int(value):
        return int(value)
    return default or Null

def is_int(value):
    try:
        int(value)
    except ValueError:
        return False
    except TypeError:
        return False
    return True

def maybe_float(value, default=None):
    if is_float(value):
        return float(value)
    return default or Null

def is_float(value):
    try:
        float(value)
    except ValueError:
        return False
    except TypeError:
        return False
    return True

@curry
def short_circuit(function, value):
    if not function(value):
        return Null
    return value

@curry
def only_if_key(key, func, d):
    if key in d:
        return func(d)
    return d

def first(seq):
    if seq:
        return _first(seq)
    return Null
maybe_first = first

def first_true(seq):
    for v in seq:
        if v:
            return v
    return Null

def second(seq):
    if seq:
        return _second(seq)
    return Null

def last(seq):
    if seq:
        return _last(seq)
    return Null

_sorted = builtins.sorted
@curry
def sort_by(func, iterable, **kw):
    return _sorted(iterable, key=func, **kw)

@curry
def sorted(iterable, **kw):
    return _sorted(iterable, **kw)

@curry
def get(i, indexable, default=None):
    if hasattr(indexable, 'get'):
        return indexable.get(i, default)
    return _get(i, indexable, default)

@curry
def getmany(keys, indexable, default=None):
    for k in keys:
        yield get(k, indexable, default=default)

_getattr = builtins.getattr
@curry
def getattr(attr, obj):
    return _getattr(obj, attr)

@curry
def vdo(func, value):
    return do(vcall(func), value)

def do_nothing(value):
    return value

def cmerge(*dicts):
    '''Curried merge'''
    def do_merge(*more_dicts):
        return merge(*(dicts + more_dicts))
    return do_merge

@curry
def create_key(key, value_function, d):
    '''Create key if it doesn't already exist'''
    if key not in d:
        return assoc(d, key, value_function(d))
    return d

@curry
def update_key(key, value_function, d):
    return assoc(d, key, value_function(d))

@curry
def set_key(key, value, d):
    '''Curriable assoc'''
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

def get_slash(inet):
    return 32 - int(math.log2(ip_network(inet).num_addresses))

def is_comma_sep_ip(cs_ip):
    return ',' in cs_ip and all(is_ip(v) for v in cs_ip.split(','))

def is_ip_range(ip_range):
    if '-' in ip_range:
        parts = ip_range.split('-')
        if len(parts) == 2:
            base, last = parts
            if is_ipv4(base) and last.isdigit() and (0 <= int(last) <= 255):
                return True
    return False

def ip_to_seq(ip):
    if is_ip(ip):
        return [ip]
    elif is_network(ip):
        return pipe(ip_network(ip).hosts(), map(str), tuple)
    elif is_interface(ip):
        return pipe(ip_interface(ip).network.hosts(), map(str), tuple)
    elif is_comma_sep_ip(ip):
        return ip.split(',')
    elif is_ip_range(ip):
        base, last = ip.split('-')
        base = ip_address(base)
        last = int(last)
        first = int(str(base).split('.')[-1])
        return [str(ip_address(int(base) + i))
                for i in range(last - first + 1)]
    else:
        log.error(f'Unknown/unparsable ip value: {ip}')
        return []

def sortips(ips):
    return sort_by(compose(ip_address, strip, strip_comments), ips)

def get_ips_from_file(path):
    return get_ips_from_str(Path(path).read_text())

def get_ips_from_str(content):
    return get_ips_from_lines(content.splitlines())

def get_ips_from_lines(lines):
    return pipe(
        lines,
        strip_comments,
        filter(lambda l: l.strip()),
        mapcat(ip_re.findall),
        # filter(is_ip),
        # mapcat(ip_to_seq),
        tuple,
    )

@curry
def in_ip_range(ip0, ip1, ip):
    start = int(ip_address(ip0))
    stop = int(ip_address(ip1))
    return int(ip_address(ip)) in range(start, stop + 1)

def is_str(v):
    return isinstance(v, str)
is_not_string = complement(is_str)

def is_dict(d):
    return isinstance(d, collections.abc.Mapping)
    # if isinstance(d, (dict, PMap, CommentedMap, OrderedDict)):
    #     return True
    # return False
is_not_dict = complement(is_dict)

def is_indexable(s):
    return hasattr(s, '__getitem__')

def is_seq(s):
    return (isinstance(s, collections.abc.Iterable) and not
            is_dict(s) and not
            isinstance(s, (str, bytes)))
    # if isinstance(s, (list, tuple, PVector, CommentedSeq)):
    #     return True
    # return False
is_not_seq = complement(is_seq)

def to_pyrsistent(obj):
    # return pyrsistent.freeze(obj)
    if is_dict(obj):
        return pipe(
            obj.items(),
            vmap(lambda k, v: (k, to_pyrsistent(v))),
            pmap,
        )
    if is_seq(obj):
        return pipe(obj, map(to_pyrsistent), pvector)
    return obj

def no_pyrsistent(obj):
    # return pyrsistent.thaw(obj)
    if is_dict(obj):
        return pipe(
            obj.items(),
            vmap(lambda k, v: (k, no_pyrsistent(v))),
            dict,
        )
    if is_seq(obj):
        return pipe(obj, map(no_pyrsistent), tuple)
    return obj

def flatdict(obj, keys=()):
    if is_dict(obj):
        for k, v in obj.items():
            yield from flatdict(v, keys + (k, ))
    else:
        yield keys + (obj,)

def ctime(path: anypath):
    return Path(path).stat().st_ctime

def dt_ctime(path: anypath):
    return pipe(
        path,
        ctime,
        datetime.fromtimestamp,
    )

def to_dt(v, default=datetime.fromtimestamp(0)):
    try_except = [
        (lambda v: dateutil.parser.parse(v), (ValueError, TypeError)),
        (lambda v: datetime.strptime(v, "%Y%m%dT%H%M%S%f"),
         (ValueError, TypeError)),
    ]
    for func, excepts in try_except:
        try:
            output = func(v)
            return output
        except excepts:
            continue
    return default

@curry
def newer(path: anypath, test: anypath):
    '''Is the path newer than the test path?'''
    return ctime(path) > ctime(test)

@curry
def older(path: anypath, test: anypath):
    '''Is the path older than the test path?'''
    return ctime(path) < ctime(test)

def function_from_path(func_path: str):
    '''Return the function object for a given module path

    '''
    return pipe(
        func_path,
        lambda path: path.rsplit('.', 1),
        vcall(lambda mod_path, func_name: (
            importlib.import_module(mod_path), func_name
        )),
        vcall(lambda mod, name: ((name, _getattr(mod, name))
                                 if hasattr(mod, name) else (name, None))),
    )

def walk(path):
    return pipe(
        os.walk(path),
        vmapcat(lambda root, dirs, files: [Path(root, f) for f in files]),
    )

@curry
def walkmap(func, root):
    return pipe(
        walk(root),
        map(func),
    )

def freeze(func):
    '''Ensure output of func is immutable

    Uses pyrsistent.freeze() on the output of func
    '''
    @functools.wraps(func)
    def return_frozen(*a, **kw):
        return pipe(func(*a, **kw), to_pyrsistent)
    return return_frozen

frozen_curry = compose(curry, freeze)

def to_paths(*paths):
    return pipe(paths, map(Path), tuple)

@curry
def log_lines(log_function, lines):
    return pipe(
        lines,
        mapcat(lambda line: line.splitlines()),
        filter(None),
        map(log_function),
    )

@curry
def grep(raw_regex, iterable, **kw):
    regex = re.compile(raw_regex, **kw)
    return filter(lambda s: regex.search(s), iterable)

@curry
def grepv(raw_regex, iterable, **kw):
    regex = re.compile(raw_regex, **kw)
    return filter(lambda s: not regex.search(s), iterable)

@curry
def grepitems(raw_regex, iterable, **kw):
    regex = re.compile(raw_regex, **kw)
    return pipe(
        iterable,
        filter(lambda items: any(regex.search(s) for s in items)),
        tuple,
    )

@curry
def grepvitems(raw_regex, iterable, **kw):
    regex = re.compile(raw_regex, **kw)
    return pipe(
        iterable,
        filter(lambda items: not any(regex.search(s) for s in items)),
        tuple,
    )

def zpad(ip):
    return '.'.join(s.zfill(3) for s in str(ip).strip().split('.'))

def unzpad(ip):
    return pipe(ip.split('.'), map(int), map(str), '.'.join)

# zpad = unzpad

def check_parents_for_file(name, start_dir=Path('.')):
    start_path = Path(start_dir).expanduser().resolve()
    directories = concatv([start_path], start_path.parents)
    for base in directories:
        path = Path(base, name)
        if path.exists():
            log.warn(
                f'Found {path} as possible site YAML config.'
            )
            return path

def shuffled(seq):
    tup = tuple(seq)
    return random.sample(tup, len(tup))

@curry
def random_sample(N, seq):
    return random.sample(tuple(seq), N)

SAM_RE = re.compile(
    r'^(.*?):\d+:(\w+:\w+):::$', re.M,
)

def get_sam_hashes(content):
    return SAM_RE.findall(content)

MSCACHE_RE = re.compile(
    r'^(.+?)/(.+?):(\$.*?\$.*?#.*?#.*?)$', re.M,
)

def get_mscache_hashes(content):
    return MSCACHE_RE.findall(content)

def strip(content):
    return content.strip()

@dispatch(str)
def strip_comments(line):
    return line[:line.index('#')] if '#' in line else line
    
@dispatch((list, tuple, PVector))  # noqa
def strip_comments(lines):
    return pipe(
        lines,
        map(strip_comments),
        tuple,
    )

def remove_comments(lines):
    return pipe(
        lines,
        filter(lambda l: not l.startswith('#')),
    )

def xlsx_pbcopy(content):
    if not content.endswith('\n'):
        content += '\n'
    pyperclip.copy(content)

def get_content(inpath, clipboard=False):
    if inpath:
        content = Path(inpath).read_text()
    elif clipboard:
        content = pyperclip.paste()
    else:
        content = sys.stdin.read()
    return content

@memoize
def nmap_services(path='/usr/share/nmap/nmap-services'):
    return pipe(
        Path(path).read_text().splitlines(),
        strip_comments,
        filter(None),
        map(lambda l: l.split('\t')[:3]),
        map(lambda t: tuple(
            concatv(t[:1], t[1].split('/'), map(float, t[-1:]))
        )),
        sort_by(lambda t: t[-1]),
        vmap(lambda name, port, proto, perc: {
            'name': name, 'port': port, 'proto': proto, 'perc': perc,
        }),
        tuple,
    )
    

@curry
def top_ports(n, *, proto='tcp', services_generator=nmap_services,
              just_ports=True):
    '''For a given protocol ('tcp' or 'udp') and a services generator
    (default nmap services file), return the top n ports

    '''
    return pipe(
        services_generator(),
        groupby(lambda d: d['proto']),
        lambda d: d[proto],
        sort_by(lambda d: d['perc'], reverse=True),
        map(get('port')) if just_ports else do_nothing,
        take(n),
        tuple,
    )

@memoize
def user_agents():
    return data_path('user-agents.txt').read_text().splitlines()

def random_user_agent():
    return pipe(
        user_agents(),
        random.choice,
    )

@curry
@functools.wraps(json.dumps)
def json_dumps(*a, **kw):
    return json.dumps(*a, **kw)

def difflines(A, B):
    linesA = pipe(
        A.splitlines(),
        strip_comments,
        filter(None),
        set,
    )
    linesB = pipe(
        B.splitlines(),
        strip_comments,
        filter(None),
        set,
    )
    return pipe(linesA - linesB, sorted)

def intlines(A, B):
    linesA = pipe(
        A.splitlines(),
        strip_comments,
        filter(None),
        set,
    )
    linesB = pipe(
        B.splitlines(),
        strip_comments,
        filter(None),
        set,
    )
    return pipe(linesA & linesB, sorted)

def escape_row(row):
    return pipe(
        row,
        map(lambda v: v.replace('"', '""')),
        '\t'.join,
        # lambda s: f'"{s}"',
    )

def output_rows_to_clipboard(rows):
    import pyperclip
    return pipe(
        rows,
        map(escape_row),
        '\n'.join,
        pyperclip.copy,
    )

@curry
def peek(nbytes, path):
    with path.open('r', encoding='latin-1') as rfp:
        data = rfp.read(nbytes)
    return data

def backup_path(path):
    path = Path(path)
    dt = dt_ctime(path)
    return Path(
        path.parent,
        ''.join((
            'backup',
            '-',
            path.stem,
            '-',
            dt.strftime('%Y-%m-%d_%H%M%S'),
            path.suffix
        ))
    )
    
def arg_intersection(func, kw):
    params = inspect.signature(func).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    else:
        return {k: kw[k] for k in set(params) & set(kw)}

def positional_only_args(func):
    return pipe(
        inspect.signature(func).parameters.values(),
        filter(
            lambda p: p.kind not in {p.VAR_KEYWORD,
                                     p.KEYWORD_ONLY,
                                     p.VAR_POSITIONAL}
        ),
        filter(lambda p: p.default == p.empty),
        map(lambda p: p.name),
        tuple,
    )

def is_arg_superset(kwargs, func):
    '''Does the kwargs dictionary contain the func's required params? '''
    return pipe(
        func,
        positional_only_args,
        set(kwargs).issuperset,
    )

@curry
def valmaprec(func, d, **kw):
    if is_dict(d):
        return pipe(
            d.items(),
            vmap(lambda k, v: (k, valmaprec(func, v, **kw))),
            type(d),
        )
    elif is_seq(d):
        return pipe(
            d, map(valmaprec(func, **kw)), type(d),
        )
    else:
        return func(d)

@curry
def regex_transform(regexes, text):
    '''Given a sequence of [(regex, replacement_text), ...] pairs,
    transform text by making all replacements

    '''
    if not is_str(text):
        return text

    regexes = pipe(
        regexes,
        vmap(lambda regex, replace: (re.compile(regex), replace)),
        tuple,
    )
    for regex, replace in regexes:
        text = regex.sub(replace, text)
    return text

def first(seq):
    if seq:
        return _first(seq)
    return Null()

@curry
def seti(index, func, seq):
    '''Return iterable of seq with value at index modified by func'''
    for i, v in enumerate(iter(seq)):
        if i == index:
            yield func(v)
        else:
            yield v
seti_t = compose(tuple, seti)

@curry
def vseti(index, func, seq):
    '''Return iterable of seq with value at index modified by func'''
    for i, v in enumerate(iter(seq)):
        if i == index:
            yield func(*v)
        else:
            yield v
vseti_t = compose(tuple, vseti)

@curry
def update_if_key_exists(key, value_function, d):
    '''Update key only if it already exist'''
    if key in d:
        return assoc(d, key, value_function(d))
    return d

@curry
def update_key_v(key, value_function, d, default=None):
    return assoc(d, key, value_function(d.get(key, default)))

@curry
def from_edgelist(edgelist, factory=None):
    '''Curried nx.from_edgelist'''
    return nx.from_edgelist(edgelist, create_using=factory)

@curry
def bfs_tree(G, source, reverse=False, depth_limit=None):
    return nx.traversal.bfs_tree(G, source, reverse=reverse,
                                 depth_limit=depth_limit)

@curry
def contains(value, obj):
    return value in obj
