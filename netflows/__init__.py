"""
netflows
========

netflows is a Python package for optimal traffic assignment / Wardrop Equilibrium on brain or transportation networks.

Source::

    https://github.com/yingqiuz/netflows
    
Simple example
---------------------
TBC

License
---------------------
TBC
"""

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    s = "Python 2.7 or later is required for netflows (%d.%d detected). "
    raise ImportError(m % sys.version_info)
del sys

# release data
# TBC

from netflows.Graph import Graph
import netflows.funcs


