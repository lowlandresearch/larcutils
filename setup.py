import os
from pathlib import Path

from setuptools import setup, find_packages

HERE = Path(__file__).resolve().parent

def version():
    return Path(HERE, 'VERSION').read_text()

# long_description = Path(HERE,'README.rst').resolve().read_text()
long_description = Path(HERE,'README.md').resolve().read_text()

def all_ext(path, ext):
    for root, dirs, files in os.walk(path):
        if any(n.endswith('.coco') for n in files):
            yield str(Path(root, '*.coco'))

setup(
    name='larcutils',
    packages=find_packages(
        exclude=['config', 'tests'],
    ),
    package_dir={
        'larcutils': 'larcutils',
    },

    # package_data={
    #     'larcutils': [
    #     ],
    # },
    # include_package_data=True,

    install_requires=Path(
        HERE, 'requirements.txt'
    ).read_text().strip().splitlines(),

    version=version(),
    description=('Collection of helper functions and general utilities'
                 ' used across various LARC repositories'),
    long_description=long_description,

    url='https://github.org/lowlandresearch/larcutils',

    author='Lowland Applied Research Company (LARC)',
    author_email='dogwynn@lowlandresearch.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],

    zip_safe=False,

    keywords=('utilities functional toolz'),

    scripts=[
    ],

    entry_points={
        'console_scripts': [
        ],
    },
)
