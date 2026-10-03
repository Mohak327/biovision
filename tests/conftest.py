import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from biovision import io
from biovision.core.field import VisualField

SIZE = 32


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def sample():
    """A bundled photograph, (height, width, 3) in [0, 1]."""
    return io.load_sample("astronaut")


@pytest.fixture(scope="session")
def field():
    return VisualField(SIZE, 60.0)
