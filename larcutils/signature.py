from collections import namedtuple
import platform
import json
import encodings
import hashlib
import socket
import logging

from toolz.curried import (
    compose, pipe, map, filter, first, partial, curry, concatv, take,
    merge, dissoc,
)
from .common import (
    vcall,
)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_sig_attr = ['hostname', 'platform', 'architecture', 'machine',
             'processor', 'python']
Signature = namedtuple('Signature', _sig_attr)

_py_attr = [
    'branch', 'build', 'compiler', 'implementation', 'revision', 'version'
]
Python = namedtuple('Python', _py_attr)

def host_signature():
    return Signature(
        socket.gethostname(),
        platform.platform(),
        platform.architecture(),
        platform.machine(),
        platform.processor(),
        pipe(
            _py_attr,
            map(lambda a: f'python_{a}'),
            map(lambda a: getattr(platform, a)),
            map(lambda f: f()),
            vcall(Python),
        ),
    )

def verify_signature_dict(sig):
    if not type(sig) is dict:
        log.error('Signature provided is not a dictionary')
        return False

    missing_keys = set(_sig_attr) - set(sig)
    if missing_keys:
        log.error(
            f'Signature missing keys: '
            f'{", ".join(sorted(missing_keys))}'
        )
        return False

    python = sig['python']
    missing_python_keys = set(_py_attr) - set(python)
    if missing_python_keys:
        log.error(
            f'Signature missing Python keys: '
            f'{", ".join(sorted(missing_python_keys))}'
        )
        return False

    return True

def signature_from_dict(sig):
    if not verify_signature_dict(sig):
        return None
    return Signature(**merge(
        sig, {'python': Python(**sig['python'])},
    ))
from_dict = signature_from_dict

def signature_to_dict(sig):
    return merge(sig._asdict(), {'python': dict(sig.python._asdict())})
to_dict = signature_to_dict

def signature_to_tuple(sig):
    return pipe(
        concatv(dissoc(sig._asdict(), 'python').items(),
                (('python', tuple(sig.python._asdict().items())),)),
        tuple,
    )
to_tuple = signature_to_tuple

utf8 = compose(first, encodings.utf_8.encode)

def host_signature_hash():
    return pipe(
        host_signature(),
        json.dumps,
        utf8,
        hashlib.sha512,
        lambda h: h.hexdigest(),
    )
