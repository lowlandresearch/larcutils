import logging
from pathlib import Path
from functools import wraps
from typing import Any, Union
from collections.abc import Mapping, Iterable

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap
from toolz import pipe
from multipledispatch import dispatch

from .common import no_pyrsistent

log = logging.getLogger('yaml')
log.addHandler(logging.NullHandler())

anypath = (str, Path)

@wraps(ruamel.yaml.dump)
def dump(*a, **kw) -> str:
    kw['Dumper'] = ruamel.yaml.RoundTripDumper
    kw['default_flow_style'] = False
    kw['width'] = 2**31
    return ruamel.yaml.dump(*a, **kw)

@wraps(ruamel.yaml.load)
def load(*a, **kw) -> Any:
    kw['Loader'] = ruamel.yaml.RoundTripLoader
    return ruamel.yaml.load(*a, **kw)
    
def read_yaml(path: anypath):
    with Path(path).open() as rfp:
        data = load(rfp)
    return data

@dispatch(anypath, Mapping)
def write_yaml(path, dict_data):
    '''Write data as YAML to path

    Args:
      path (str, Path): path to write to

      data (dict-like): dictionary-like object to write (will be
         recursively converted to base Python types)

    Returns: (bool) success of write operation,

    Raises: on error, will raise exception

    '''
    return _write_yaml(path, pipe(dict_data, no_pyrsistent, CommentedMap))

@dispatch(anypath, Iterable)    # noqa
def write_yaml(path, data):
    '''Write data as YAML to path

    Args:
      path (str, Path): path to write to

      data (sequence): sequence object to write (will be recursively
         converted to base Python types)

    Returns: (bool) success of write operation,

    Raises: on error, will raise exception

    '''
    return _write_yaml(path, pipe(data, no_pyrsistent))

@dispatch(anypath, object)      # noqa
def write_yaml(path, data):
    return _write_yaml(path, data)

def _write_yaml(path, data):
    with Path(path).open('w') as wfp:
        dump(data, wfp)
    return True
