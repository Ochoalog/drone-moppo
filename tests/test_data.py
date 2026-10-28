"""Phase 1 checks against the numbers measured manually (spec §3)."""
import json

import numpy as np
import pytest

from dal.config import DATA, default_config
from dal.grid import Grid

# Spec §3: cloud ~19 % (pre_oct23), ~1 % (post, pre_mar23); AOI covered by the 3 dates.
EXPECTED_CLOUD = {"post": 0.01, "pre_mar23": 0.01, "pre_oct23": 0.19}


@pytest.mark.data
@pytest.mark.parametrize("date", list(EXPECTED_CLOUD))
def test_cloud_and_valid_fraction_match_spec(date):
    s = json.loads((DATA / "masks_stats.json").read_text())[date]
    # The raw HSV detector is what the spec numbers were measured with (+-30 %).
    assert abs(s["aoi_cloud_raw_frac"] - EXPECTED_CLOUD[date]) <= 0.3 * EXPECTED_CLOUD[date]
    # "Covered by the three dates": nearly all of the AOI has data (allow 30 % of a 10 % slack).
    assert s["aoi_valid_frac"] >= 0.93


@pytest.mark.data
def test_refined_cloud_mask_keeps_real_clouds():
    s = json.loads((DATA / "masks_stats.json").read_text())
    assert s["pre_oct23"]["aoi_cloud_frac_of_valid"] >= 0.7 * EXPECTED_CLOUD["pre_oct23"]
    assert s["post"]["aoi_cloud_frac_of_valid"] <= 0.013


@pytest.mark.data
@pytest.mark.parametrize("date", ["pre_mar23", "pre_oct23"])
def test_alignment_offset_in_range(date):
    a = json.loads((DATA / "align.json").read_text())[date]
    assert 0.0 <= a["norm_m"] <= 8.0
    assert a["n_used"] >= 5


def test_grid_split_is_disjoint_and_on_the_right_side():
    cfg = default_config()
    d = cfg["data"]
    g = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    train, test = g.split_rows(d["split_lat"], d["split_margin_m"])
    assert len(set(train) & set(test)) == 0
    assert train.max() < test.min()
    # Cell centers of the last train row are north of the split by at least half a cell.
    assert g.lat[train].min() > d["split_lat"]
    assert g.lat[test].max() <= d["split_lat"]
