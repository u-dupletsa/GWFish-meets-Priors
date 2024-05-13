"""
GWFish + Priors
=====

Prior-informed GWFish

"""

import sys

import priors, tools, minimax_tilting_sampler

if sys.version_info < (3,):
    raise ImportError("You need Python 3")
