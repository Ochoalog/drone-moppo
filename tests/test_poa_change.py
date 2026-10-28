import numpy as np

from dal.poa import change_index


def test_change_index_range_and_zero():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(10, 12, 16))
    o = rng.normal(size=(2, 10, 12, 16))
    ch = change_index(m, o)
    assert ch.shape == (10, 12) and ch.min() >= 0 and ch.max() <= 2
    assert np.allclose(change_index(m, np.stack([m, m])), 0, atol=1e-6)
    assert np.allclose(change_index(m, -np.stack([m, m])), 2, atol=1e-6)
