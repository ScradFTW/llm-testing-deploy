import os
import sys

import joblib

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.dirname(_TESTS_DIR)
_SERVE_DIR = os.path.join(_SERVICE_DIR, "serve")
_REPO_MODEL_PATH = os.path.join(_SERVICE_DIR, "train", "genre_pipeline.joblib")
_CONTAINER_MODEL_PATH = "/opt/genre-classifier/genre_pipeline.joblib"

if _SERVE_DIR not in sys.path:
    sys.path.insert(0, _SERVE_DIR)

# app.py hardcodes MODEL_PATH to /opt/genre-classifier/genre_pipeline.joblib,
# which is only populated by the Dockerfile inside the built image. Redirect
# joblib.load, for that one path only, to the model file committed at
# genre-classifier/train/genre_pipeline.joblib, so importing app.py in tests
# exercises the real trained pipeline without requiring root/`/opt` access
# or any change to app.py itself.
_real_joblib_load = joblib.load


def _load_redirecting_container_path(path, *args, **kwargs):
    if path == _CONTAINER_MODEL_PATH:
        path = _REPO_MODEL_PATH
    return _real_joblib_load(path, *args, **kwargs)


joblib.load = _load_redirecting_container_path
