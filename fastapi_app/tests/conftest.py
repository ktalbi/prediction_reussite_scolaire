import os
import sys
import importlib
import pytest
from fastapi.testclient import TestClient


def _import_app():

    # 1) Essayer import direct (si main.py est importable)
    try:
        mod = importlib.import_module("main")
        return mod.app
    except Exception:
        pass

    # 2) Essayer import package (si fastapi_app est un package)
    try:
        mod = importlib.import_module("fastapi_app.main")
        return mod.app
    except Exception:
        pass

    # 3) Dernier recours : ajuster sys.path
    here = os.path.dirname(__file__)
    fastapi_app_dir = os.path.abspath(os.path.join(here, ".."))      # .../fastapi_app
    repo_root = os.path.abspath(os.path.join(fastapi_app_dir, "..")) # repo root
    for p in (repo_root, fastapi_app_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        mod = importlib.import_module("main")
        return mod.app
    except Exception:
        mod = importlib.import_module("fastapi_app.main")
        return mod.app


@pytest.fixture(scope="module")
def client():
    app = _import_app()
    return TestClient(app)
