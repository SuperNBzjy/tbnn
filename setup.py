###############################################################
#
# Copyright 2017 Sandia Corporation. Under the terms of
# Contract DE-AC04-94AL85000 with Sandia Corporation, the
# U.S. Government retains certain rights in this software.
# This software is distributed under the BSD-3-Clause license.
#
##############################################################

import sys
import os
import codecs

# to create the distribution:  python setup.py sdist
# to install the distribution: python setup.py install

if sys.version_info[:2] < (3, 0):
    print('WARNING: TBNN now targets Python 3. Your version of Python ({0}.{1}.{2}) is unsupported.'
          .format(*sys.version_info) + os.linesep +
          '         You may experience instability with this build.')
if os.name != 'posix':
    print('WARNING: TBNN has not been tested with your operating system, ({0}).'.format(os.name) + os.linesep +
          '         You may experience some unstability with this build.')

metadata = dict()
exclude = ['tests', 'examples']
try:
    from setuptools import setup, find_packages
except ImportError as exc:  # pragma: no cover - setuptools is expected to be available
    raise ImportError(
        'setuptools is required to install tbnn. Please install it and retry.'
    ) from exc

metadata['packages'] = find_packages(exclude=[x + '.*' for x in exclude])

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'DESCRIPTION.txt'), encoding='utf-8') as f:
    long_description = f.read()

version_ns = {}
with open(os.path.join(here, 'tbnn', 'version.py')) as f:
    exec(f.read(), version_ns)
__version__ = version_ns['__version__']

metadata.update({
    'name': 'tbnn',
    'version': __version__,
    'description': 'A Tensor-Basis Neural Network Trainer',
    'long_description': long_description,
    'author': 'Julia Ling',
    'author_email': 'jling@sandia.gov',
    'maintainer': 'Jeremy Templeton',
    'maintainer_email': 'jatempl@sandia.gov',
    'install_requires': ['lasagne==0.2.dev1',
                         'theano>=0.7.0.dev',
                         'matplotlib>=1.5.1',
                         'numpy>=1.6.2',
                         'scipy>=0.11',
    ],
    'classifiers': [
            'Development Status :: 2 - Pre-Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3'
    ],
    'url': 'http://www.github.com/tbnn',
    'keywords': 'sandia neuralnetwork tensor',
    'test_suite': 'nose.collector',
    'tests_require': ['nose']
})

setup(**metadata)

