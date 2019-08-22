from pathlib import Path
import urllib
from dataclasses import dataclass
import functools
import inspect
import re

import requests
from toolz import (
    pipe, curry, partial, concat, concatv, compose,
)
from toolz.curried import (
    map, filter, first, last, merge,
)
from pyrsistent import pmap

from larcutils.common import (
    maybe_pipe, vmap, vfilter, maybe_first, maybe_int,
)

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

        for method in ['get', 'post', 'put', 'delete', 'head',
                       'options', 'patch']:
            setattr(
                self, method,
                partial(
                    getattr(self.api.session, method), **self.kwargs
                ),
            )

    def __call__(self, *parts, **kw):
        return Endpoint(
            self.api, tuple(concatv(self.parts, parts)), **kw
        )

    @property
    def url(self):
        return urllib.parse.urljoin(
            self.api.base_url, '/'.join(pipe(self.parts, map(str)))
        )

    @curry
    def iter(self, method, url=None, iter_f=lambda r: r.json(),
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

        for value in iter_f(response):
            yield value

        next_url = link_next(response)
        if next_url:
            yield from self.iter(
                method, url=next_url, iter_f=iter_f, **requests_kw
            )

