"""
GWFish + Priors
=====

Prior-informed GWFish

"""

import sys

from . import priors, tools

if sys.version_info < (3,):
    raise ImportError("You need Python 3")
