from pathlib import Path
import urllib
from dataclasses import dataclass
import functools
import inspect
import re
import logging

import requests
from toolz import (
    pipe, curry, partial, concat, concatv, compose,
)
from toolz.curried import (
    map, filter, first, last, merge, do,
)
from pyrsistent import pmap

from larcutils.common import (
    maybe_pipe, vmap, vfilter, maybe_first, maybe_int, do_nothing,
    is_dict, vcall,
)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class ResponseError(ValueError):
    pass

class TokenAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, request):
        request.headers['Authorization'] = 'Bearer {}'.format(self.token)
        return request

class Api:
    def __init__(self, base_url, auth, session):
        self.base_url = base_url
        self.auth = auth
        self.session = session

    def __call__(self, *parts, **kw):
        return Endpoint(
            self, parts, **kw
        )

def link_next(response):
    return maybe_pipe(
        requests.utils.parse_header_links(
            response.headers.get('Link', '')
        ),
        filter(lambda d: d.get('rel', '').lower() == 'next'),
        maybe_first,
        lambda d: d['url'],
    )

def method_logger(name, method):
    @functools.wraps(method)
    def wrapper(*a, **kw):
        log.debug(f'{name.upper()}: {kw}')
        return method(*a, **kw)
    return wrapper

def namespace_data(data):
    def ns_dict(ns, d):
        return pipe(
            d.items(),
            vmap(lambda k, v: (f'{ns}[{k}]', v)),
            dict,
        )
    if is_dict(data):
        return pipe(
            data.items(),
            vmap(lambda k, v: ns_dict(k, v) if is_dict(v) else {k: v}),
            vcall(merge),
        )
    return data
                    
class Endpoint:
    def __init__(self, api, parts, **kwargs):
        self.api = api
        self.parts = tuple(parts)

        self.kwargs = pipe(
            merge(
                {'url': self.url, 'auth': self.api.auth},
                kwargs,
            ),
            pmap,
        )

        for name in ['get', 'post', 'put', 'delete', 'head',
                     'options', 'patch']:
            setattr(
                self, name, self.method(name, **self.kwargs)
            )

    def __call__(self, *parts, **kw):
        return Endpoint(
            self.api, tuple(concatv(self.parts, parts)), **kw
        )

    def method(self, name, **orig_kw):
        def caller(*a, **kw):
            method = getattr(self.api.session, name)
            kw = merge(orig_kw, kw)
            if 'data' in kw:
                kw['data'] = pipe(
                    kw.pop('data'),
                    namespace_data,
                )
            return method_logger(name, method)(*a, **kw)
        return caller

    @property
    def url(self):
        base = self.api.base_url
        if not base.endswith('/'):
            base += '/'
        return urllib.parse.urljoin(
            base, '/'.join(pipe(self.parts, map(str)))
        )

    @curry
    def iter(self, method, url=None,
             iter_f=lambda resp: resp.json(),
             link_next_f=link_next, **requests_kw):
        response = getattr(self, method)(**(
            merge(requests_kw, {'url': url} if url else {})
        ))
        if response.status_code != 200:
            content = response.content.decode()
            raise ResponseError(
                f'Response code error: {response.status_code}\n\n'
                f'{content[:200]}\n'
                '...\n'
                '...\n'
                f'{content[-200:]}'
            )

        for value in iter_f(self, response):
            yield value

        next_url = link_next(response)
        if next_url:
            yield from self.iter(
                method, url=next_url, iter_f=iter_f, **requests_kw
            )

class ResourceEndpoint(Endpoint):
    def __init__(self, endpoint: Endpoint, data: dict, form_key: str):
        self.data = data
        self.form_key = form_key

        super().__init__(endpoint.api, endpoint.parts)
        
class IdResourceEndpoint(Endpoint):
    def __init__(self, parent: Endpoint, data: dict, form_key: str,
                 id_key: str):
        self.parent = parent
        self.data = data
        self.form_key = form_key
        self.id_key = id_key

        if id_key not in data:
            log.error(f'ID key {id_key} not in data: {data}')

        super().__init__(parent.api, parent.parts + (data[id_key],))

    @classmethod
    @curry
    def from_multiple_response(cls, parent: Endpoint,
                               response: requests.Response, *,
                               form_key=None, id_key='id',
                               unpack_f=do_nothing):
        for d in unpack_f(response.json()):
            yield cls(parent, d, form_key, id_key)

    @classmethod
    @curry
    def from_single_response(cls, parent: Endpoint,
                             response: requests.Response, *,
                             form_key=None, id_key='id',
                             unpack_f=do_nothing):
        data = unpack_f(response.json())
        return cls(parent, data, form_key, id_key)

    def refresh(self, **get_kw):
        return IdResourceEndpoint(
            self.parent, self.get(**get_kw).json(),
            self.form_key, self.id_key,
        )

@curry
def update_endpoint(endpoint: IdResourceEndpoint, update: dict, *,
                    get_kw=None, do_refresh=True):
    endpoint.put(data=({endpoint.form_key: update}
                       if endpoint.form_key is not None else update))
    if do_refresh:
        return endpoint.refresh(**(get_kw or {}))

_cache = {}
def memoize_key(resource_name: str, endpoint: Endpoint):
    return (resource_name, endpoint.url)

@curry
def memoize_resource(resource_name: str, endpoint: Endpoint, data):
    _cache[memoize_key(resource_name, endpoint)] = data
    return data
    
def reset_cache():
    global _cache
    _cache = {}
def reset_cache_by_key(name, endpoint):
    global _cache
    key = memoize_key(name, endpoint)
    if key in _cache:
        del _cache[key]

def get_id_resource(name: str, id: (int, str), *, form_key: str = None,
                    id_key: str = 'id', unpack_f=do_nothing,
                    help=None, **iter_kw):
    def getter(endpoint: IdResourceEndpoint):
        return pipe(
            endpoint(name, id).get(),
            IdResourceEndpoint.from_single_response(
                form_key=form_key, id_key=id_key, unpack_f=unpack_f,
            )
        )
    getter.__doc__ = help or ''

    getter.reset_cache = reset_cache
    getter.reset_cache_by_key = reset_cache_by_key

    return getter

def get_id_resources(name: str, *, form_key: str = None, id_key: str = 'id',
                     unpack_f=do_nothing, help=None, memo=False,
                     memo_resource_name=None, **iter_kw):
    def getter(endpoint: IdResourceEndpoint):
        key = memoize_key(memo_resource_name or name, endpoint)
        if memo and key in _cache:
            return _cache[key]

        return pipe(
            endpoint(name).iter(
                'get', **merge(
                    {'iter_f': compose(
                        IdResourceEndpoint.from_multiple_response(
                            form_key=form_key, id_key=id_key,
                            unpack_f=unpack_f,
                        ),
                    )},
                    iter_kw
                )
            ),
            tuple,
            memoize_resource(name, endpoint),
        )
    getter.__doc__ = help or ''

    getter.reset_cache = reset_cache
    getter.reset_cache_by_key = reset_cache_by_key

    return getter

def new_id_resource(name: str, *,
                    form_key: str = None, id_key: str = 'id',
                    get_kw=None, help: str = None, memo=False,
                    memo_resource_name=None):
    def creator(endpoint: Endpoint, data: dict):
        data = {form_key: data} if form_key is not None else data
        response = endpoint(name).post(data=data)
        if response.status_code in range(200, 300):
            post_data = response.json()
            new_response = endpoint(
                name, post_data[id_key]
            ).get(**(get_kw or {}))
            if new_response.status_code in range(200, 300):
                key = memoize_key(memo_resource_name or name, endpoint(name))
                if key in _cache:
                    del _cache[key]
                return IdResourceEndpoint(
                    endpoint(name), new_response.json(),
                    form_key=form_key, id_key=id_key,
                )
            log.error(
                f'There was an error after creating {name} object:\n'
                f'  data: {data}\n'
                f'  form_key: {form_key}\n'
                f'  id_key: {id_key}\n'
                'Response:\n'
                f'{new_response.content[:1000]}'
            )
        log.error(
            f'There was an error creating {name} object:\n'
            f'  data: {data}\n'
            f'  form_key: {form_key}\n'
            f'  id_key: {id_key}\n'
            'Response:\n'
            f'{response.content[:1000]}'
        )
    creator.__doc__ = help or ''
    return creator

