"""Test configuration: hermetic, offline, no model downloads or cloud services."""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="roomspec-test-"))
os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "QDRANT_URL": "",
        "EMBEDDING_BACKEND": "hash",
        "LLM_PROVIDER": "none",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
        "AUTO_SEED": "true",
        "LOG_LEVEL": "WARNING",
    }
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def _encode(img: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def room_jpeg() -> bytes:
    from make_samples import SCENES, render

    return _encode(render(*SCENES["warm_walnut_kitchen"], seed=1))


@pytest.fixture
def encode():
    return _encode
