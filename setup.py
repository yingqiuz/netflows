#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for netflows

you can install netflows with

python setup.py install
"""

import os
import sys
import setuptools


if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()

if sys.version_info[:2] < (3, 5):
    print("netflows requires Python 3.5 or later (%d.%d detected)." %
          sys.version_info[:2])
    sys.exit(-1)

# write the version information
sys.path.insert(0, 'netflows')
import info
sys.path.pop(0)

# get long description from README
curr_path = os.path.dirname(__file__)
with open(os.path.join(curr_path, info.__longdesc__), "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":

    setuptools.setup(
        name=info.__packagename__,
        version=info.__version__,
        author=info.__author__,
        author_email=info.__email__,
        description=info.__description__,
        long_description=long_description,
        long_description_content_type=info.__longdesctype__,
        url=info.__url__,
        packages=setuptools.find_packages(exclude=['netflows/tests']),
        classifiers=info.CLASSIFIERS,
        install_requires=info.REQUIRES,
        extras_requires=info.EXTRAS_REQUIRE,
        test_requires=info.TESTS_REQUIRE,
        license=info.__license__,
    )
