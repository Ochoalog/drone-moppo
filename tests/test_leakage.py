"""No train window may touch lat <= split (real grid); the train/test split is disjoint."""
import numpy as np
import pytest

from dal.config import default_config
from dal.grid import Grid, utm_to_lonlat


@pytest.mark.data
@pytest.mark.parametrize("scenario", ["S0", "S1", "S2"])
def test_train_windows_never_touch_south(scenario):
    from dal.env import ScenarioData

    cfg = default_config()
    d = cfg["data"]
    g = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    data = ScenarioData.load(cfg["env"]["descriptor"], scenario, cfg)
    W = cfg["filter"]["window"]
    margin = cfg["features"]["footprint_m"] / 2 + cfg["features"]["jitter_m"]
    wins = data.windows("train", W, cfg["filter"]["min_window_valid"])
    assert len(wins) > 0
    r_last = wins[:, 0].max() + W - 1
    # southern-most point any training footprint can see, over the whole width
    y = g.y1 - (r_last + 0.5) * g.cell - margin
    xs = np.linspace(g.x0, g.x1, 50)
    _, lat = utm_to_lonlat(xs, np.full_like(xs, y))
    assert np.all(np.asarray(lat) > d["split_lat"])
    test = data.windows("test", W, cfg["filter"]["min_window_valid"])
    assert len(test) > 0
    assert set(range(wins[:, 0].min(), r_last + 1)).isdisjoint(range(test[:, 0].min(), test[:, 0].max() + W))
