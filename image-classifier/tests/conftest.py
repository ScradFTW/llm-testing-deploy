import builtins
import os
import sys

import numpy as np
import onnxruntime as ort

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.dirname(_TESTS_DIR)
_SERVE_DIR = os.path.join(_SERVICE_DIR, "serve")
_TRAIN_DIR = os.path.join(_SERVICE_DIR, "train")
_CONTAINER_MODEL_DIR = "/opt/image-classifier"

if _SERVE_DIR not in sys.path:
    sys.path.insert(0, _SERVE_DIR)


def _redirect(path):
    """Map a /opt/image-classifier/<file> path (only populated inside the
    built container) to the equivalent file committed under
    image-classifier/train/, so importing app.py in tests loads the real
    ONNX model and normalization stats without requiring root/`/opt`
    access or any change to app.py itself."""
    if isinstance(path, str) and path.startswith(_CONTAINER_MODEL_DIR + "/"):
        return os.path.join(_TRAIN_DIR, os.path.basename(path))
    return path


# app.py does three separate reads under MODEL_DIR at import time:
# onnxruntime.InferenceSession(...), np.load(...) (x2), and open(...) for
# classes.txt. Wrap each so only *that* specific /opt path is redirected;
# every other path (e.g. later `open()` calls elsewhere) is untouched.
_real_inference_session = ort.InferenceSession


class _InferenceSessionRedirectingContainerPath(_real_inference_session):
    def __init__(self, path_or_bytes, *args, **kwargs):
        super().__init__(_redirect(path_or_bytes), *args, **kwargs)


ort.InferenceSession = _InferenceSessionRedirectingContainerPath

_real_np_load = np.load


def _np_load_redirecting_container_path(path, *args, **kwargs):
    return _real_np_load(_redirect(path), *args, **kwargs)


np.load = _np_load_redirecting_container_path

_real_open = builtins.open


def _open_redirecting_container_path(path, *args, **kwargs):
    return _real_open(_redirect(path), *args, **kwargs)


builtins.open = _open_redirecting_container_path
