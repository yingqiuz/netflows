#!/usr/bin/env python

"""
Setup script for netflows

you can install netflows with

python setup.py install
"""

import os
import sys
from setuptools import setup, find_packages


if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()	

if sys.version_info[:2] < (3, 5):
    print("netflows requires Python 3.5 or later (%d.%d detected)." %
          sys.version_info[:2])
    sys.exit(-1)


