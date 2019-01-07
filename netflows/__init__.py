# -*- coding: utf-8 -*-
"""
netflows
========

netflows is a Python package for optimal traffic assignment /\
Wardrop Equilibrium on brain or transportation networks.

Source::

    https://github.com/yingqiuz/netflows
    
Simple example
---------------------
TBC

License
---------------------
TBC
"""

__all__ = [
    '__author__', '__description__', '__email__', '__license__',
    '__packagename__', '__url__', '__version__', 'CreateGraph',
    'wardrop_equilibrium_linear_solve', 'system_optimal_linear_solve',
    'wardrop_equilibrium_affine_solve', 'system_optimal_affine_solve',
    'wardrop_equilibrium_bpr_solve', 'system_optimal_bpr_solve'
]

from .info import (
    __version__,
    __author__,
    __description__,
    __email__,
    __license__,
    __packagename__,
    __url__
)

from .create import CreateGraph
from .funcs import (
    wardrop_equilibrium_linear_solve, system_optimal_linear_solve,
    wardrop_equilibrium_affine_solve, system_optimal_affine_solve,
    wardrop_equilibrium_bpr_solve, system_optimal_bpr_solve
)
