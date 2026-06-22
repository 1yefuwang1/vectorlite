import numpy as np
import pytest
from vectorlite_py.test.helpers import get_connection, SEED


@pytest.fixture
def conn():
    c = get_connection()
    yield c
    c.close()


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)
