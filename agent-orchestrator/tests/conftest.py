import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.dirname(_TESTS_DIR)
_SERVE_DIR = os.path.join(_SERVICE_DIR, "serve")

if _SERVE_DIR not in sys.path:
    sys.path.insert(0, _SERVE_DIR)
