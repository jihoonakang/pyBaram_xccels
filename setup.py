# -*- coding: utf-8 -*-

import re
from setuptools import setup
import sys

vfile = open('pybaram/_version.py').read()
version = re.search(r'__version__\s+=\s+\'([\d.]+)\'', vfile).group(1)

# Modules
modules = ['pybaram.backends',
           'pybaram.backends.cpu',
           'pybaram.integrators',
           'pybaram.partitions',
           'pybaram.plugins',
           'pybaram.readers',
           'pybaram.solvers',
           'pybaram.solvers.base',
           'pybaram.solvers.baseadvec',
           'pybaram.solvers.baseadvecdiff',
           'pybaram.solvers.euler',
           'pybaram.solvers.navierstokes',
           'pybaram.solvers.rans',
           'pybaram.solvers.ranssa',
           'pybaram.solvers.ranskwsst',
           'pybaram.utils',
           'pybaram.writers'
           ]

# Hard dependencies
install_requires = [
    'h5py >= 2.6',
    'mpi4py >= 2.0',
    'numpy >= 1.10',
    'numba >= 0.5',
]

# Scripts
console_scripts = [
    'pybaram = pybaram.__main__:main'
]

# Additional data
data_files = [
    ('', ['pybaram/__main__.py'])
]

setup(name='pybaram',
      version=version,
      description='Compressible CFD solver in Python',
      author='Jin Seok Park',
      packages=['pybaram'] + modules,
      data_files=data_files,
      entry_points={'console_scripts': console_scripts},
      install_requires=install_requires,
      )
