import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeEncoder  # noqa: E402


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()
