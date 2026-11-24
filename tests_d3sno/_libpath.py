"""Puts the library folder (the parent of ``tests``) first on ``sys.path``, so that the suite also runs
under ``python -I`` (isolated mode adds neither the current folder nor ``PYTHONPATH``):
``python -I -B -m unittest discover -s tests`` from the library folder. Every test module imports it
before ``d3sno``."""

import os
import sys

LIBRARY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not sys.path or os.path.abspath(sys.path[0]) != LIBRARY:
    sys.path.insert(0, LIBRARY)
