import re

import jinja2
import markdown as _markdown
# from markdown import Markdown, Extension
from markdown.preprocessors import Preprocessor
from toolz import (
    pipe, compose, curry, complement, juxt, merge,
)
from toolz.curried import (
    interleave, partition, reduce, map, do,
)

from . import yaml
from .common import vmap, vcall, maybe_pipe, short_circuit

class YamlExtension(_markdown.Extension):
    def extendMarkdown(self, md, md_globals):
        md.preprocessors.add("meta_yaml", YamlPreprocessor(md), '_end')

def monotonic(seq, cmp=lambda a, b: a <= b):
    return all(cmp(seq[i], seq[i + 1]) for i in range(0, len(seq) - 1))

is_not = complement

@curry
def first(functions, value):
    for func in functions:
        boolean, *other = func(value)
        if boolean:
            return (boolean,) + other

def fpipe(value, *functions):
    for func in functions:
        value = func(value)
        if value is None:
            return None
    return value

class MarkdownError(Exception):
    pass

start_re = re.compile(r'--yaml--', re.IGNORECASE)
end_re = re.compile(r'--endyaml--', re.IGNORECASE)

def not_enough_tags(lines):
    s, e = start_re.pattern, end_re.pattern
    return (
        lines.count(s) != lines.count(e),
        f'Start and end YAML tags do not agree: {s} start != {e} end'
    )

def start_indices(lines):
    return [i for i,l in enumerate(lines) if start_re.search(l)]

def end_indices(lines):
    return [i for i,l in enumerate(lines) if end_re.search(l)]

def all_indices(lines):
    return list(interleave(
        (start_indices(lines), end_indices(lines))
    ))

def not_correct_order(lines):
    valid = pipe(
        all_indices(lines),
        complement(monotonic),
    )
    return (
        valid, 'Start and end tags are not in the correct order.'
    )

def check_lines(lines):
    errors = [
        msg for boolean, msg in juxt(
            not_correct_order, not_enough_tags,
        )(lines) if boolean
    ]

    if errors:
        return False, '\n'.join(errors)
    return True, ''

def yaml_data(lines):
    def render(raw, data):
        return jinja2.Template(raw).render(**data)
    
    return maybe_pipe(
        all_indices(lines),
        short_circuit(bool),  # catch null YAML early
        partition(2),
        vmap(lambda s,e: lines[s + 1:e]),
        map('\n'.join),
        lambda lines: '\n'.join(lines),
        lambda raw: (raw, yaml.load(raw)),
        vcall(render),
        lambda raw: (raw, yaml.load(raw)),
        vcall(render),
        lambda data: data[1],
    )

def non_yaml_lines(lines):
    return pipe(
        [-1] + all_indices(lines) + [len(lines)],
        partition(2),
        vmap(lambda s, e: lines[s + 1:e]),
        reduce(lambda a,b: a + b)
    )

def get_metadata(content: str):
    lines = content.splitlines()
    valid, msg = check_lines(lines)
    if not valid:
        raise MarkdownError(msg)

    metadata = yaml_data(lines)

    return '\n'.join(non_yaml_lines(lines)), metadata

class YamlPreprocessor(Preprocessor):
    '''Parse out the YAML metadata blocks from the markdown

    A YAML metadata block:
    
    - Is delimited by:
        - a line `--yaml--` at the start
        - and a line `--endyaml--` at the end
    
    '''
    def run(self, lines):
        '''Parse YAML metadata blocks and store in self.markdown.metadata'''
        # print('in run')
        valid, msg = check_lines(lines)
        if not valid:
            raise MarkdownError(msg)
        
        self.markdown.metadata = yaml_data(lines)

        return non_yaml_lines(lines)


def makeExtension(**kwargs):
    return YamlExtension(**kwargs)

class Markdown(_markdown.Markdown):
    metadata = None

    def __init__(self):
        extensions = {
            'markdown.extensions.extra': {},
            'markdown.extensions.codehilite': {
                'noclasses': True,
            },
            'nsutils.markdown': {},
        }
        
        super().__init__(
            extensions=list(extensions),
            extension_configs=extensions,
        )

    def convert(self, *a, **kw):
        class StrWithMetadata(str):
            metadata = None
        html = super().convert(*a, **kw)
        html = StrWithMetadata(html)
        html.metadata = self.metadata or {}
        return html

def markdown(text, *args, **kwargs):
    """Convert a Markdown string to HTML and return HTML as a Unicode string.

    This is a shortcut function for `Markdown` class to cover the most
    basic use case.  It initializes an instance of Markdown, loads the
    necessary extensions and runs the parser on the given text.

    Keyword arguments:

    * text: Markdown formatted text as Unicode or ASCII string.
    * Any arguments accepted by the Markdown class.

    Returns: An HTML document as a string.

    """
    md = Markdown(*args, **kwargs)
    return md.convert(text)
