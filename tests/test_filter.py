import numpy as np
import pytest

from dal.filter import ACTIONS, HistogramFilter, blur, entropy, gaussian_kernel1d, shift


def _filter(W=16, D=8, invalid=(), seed=0, beta=10.0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(W, W, D)).astype(np.float32)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    ok = np.ones((W, W), bool)
    for rc in invalid:
        ok[rc] = False
    return HistogramFilter(m, ok, beta), m


def test_belief_normalized_after_update_and_predict():
    f, m = _filter(invalid=[(3, 4), (0, 0)])
    rng = np.random.default_rng(1)
    for _ in range(20):
        f.predict(int(rng.integers(8)))
        assert np.isclose(f.b.sum(), 1.0)
        f.update(m[rng.integers(16), rng.integers(16)])
        assert np.isclose(f.b.sum(), 1.0)
        assert (f.b >= 0).all()


def test_entropy_max_at_uniform_prior():
    f, _ = _filter()
    assert np.isclose(f.entropy(), np.log(16 * 16))
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random((16, 16))
        assert entropy(p / p.sum()) <= f.entropy() + 1e-9


def test_prior_uniform_over_valid_cells_only():
    f, _ = _filter(invalid=[(1, 1), (2, 2)])
    assert f.b[1, 1] == 0 and f.b[2, 2] == 0
    assert np.isclose(f.entropy(), np.log(16 * 16 - 2))


@pytest.mark.parametrize("a", range(8))
def test_shift_moves_mass_and_blocks_at_borders(a):
    dr, dc = ACTIONS[a]
    b = np.zeros((8, 8))
    b[4, 4] = 1.0
    s = shift(b, (dr, dc))
    assert s[4 + dr, 4 + dc] == 1.0
    # A delta on the border the action points to cannot move: it stays.
    r = 0 if dr < 0 else 7 if dr > 0 else 4
    c = 0 if dc < 0 else 7 if dc > 0 else 4
    b = np.zeros((8, 8))
    b[r, c] = 1.0
    assert shift(b, (dr, dc))[r, c] == 1.0


def test_shift_diagonal_blocked_on_one_axis_stays():
    b = np.zeros((8, 8))
    b[0, 3] = 1.0  # north edge, NE move is blocked entirely (no sliding east)
    s = shift(b, ACTIONS[1])
    assert s[0, 3] == 1.0 and s.sum() == 1.0


def test_shift_conserves_mass():
    b = np.random.default_rng(0).random((10, 10))
    for d in ACTIONS:
        assert np.isclose(shift(b, d).sum(), b.sum())


def test_blur_conserves_mass_and_spreads():
    b = np.zeros((9, 9))
    b[0, 0] = 1.0
    out = blur(b, gaussian_kernel1d(0.5))
    assert np.isclose(out.sum(), 1.0)
    assert out[0, 0] < 1.0 and out[1, 1] > 0


def test_likelihood_uniform_on_invalid_cells():
    f, m = _filter(invalid=[(5, 5), (7, 2)])
    L = f.likelihood(m[3, 3])
    assert np.isclose(L[5, 5], L[7, 2])
    assert np.isclose(L[5, 5], L[f.valid].mean())
    # update leaves the ratio between two invalid cells unchanged
    f.b = np.full((16, 16), 1 / 256)
    f.update(m[3, 3])
    assert np.isclose(f.b[5, 5], f.b[7, 2])


def test_invalid_observation_does_not_change_belief():
    f, m = _filter()
    f.predict(2)
    b = f.b.copy()
    f.update(m[1, 1], obs_valid=False)
    assert np.array_equal(b, f.b)


def test_true_cell_most_likely_and_concentration():
    f, m = _filter(beta=30.0)
    for _ in range(3):
        f.update(m[6, 9])
    assert f.map_estimate() == (6, 9)
    assert f.mass_around() > 0.9
