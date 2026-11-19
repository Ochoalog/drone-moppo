"""Specs v3: footprint por altitude, variantes determinísticas dada a semente e faixas do modelo do sensor."""
import math

import numpy as np
import pytest

from dal.v3.sensor import specs

SITE = specs.Site(name="fake", aoi={"lon0": -51.95, "lon1": -51.94, "lat0": -29.33, "lat1": -29.32},
                  obs_date="post", map_dates=("pre", "post"), sources={"pre": "pre.tif", "post": "post.tif"},
                  offsets={"pre": (2.5, -1.0), "post": (0.0, 0.0)})


@pytest.mark.parametrize("altitude, expected", [(60, 108.0), (100, 180.1), (150, 270.1)])
def test_footprint_follows_altitude_and_fov(altitude, expected):
    assert specs.footprint_m(altitude) == pytest.approx(expected, abs=0.1)
    assert specs.footprint_m(altitude) == pytest.approx(2 * altitude * math.tan(math.radians(42)))


def test_observation_variants_are_deterministic_given_the_seed():
    first, second = specs.obs_spec(SITE, 100), specs.obs_spec(SITE, 100)
    assert specs.content_hash(first) == specs.content_hash(second)
    assert specs.content_hash(first) != specs.content_hash(specs.obs_spec(SITE, 150))


def test_variant_parameters_stay_in_the_sensor_model_ranges():
    obs = specs.obs_spec(SITE, 60)
    grid = specs.site_grid(SITE)
    centers_x, centers_y = grid.center(obs.cell_row.to_numpy(), obs.cell_col.to_numpy())
    assert np.abs(obs.x - centers_x).max() <= specs.JITTER_M + 1e-9
    assert np.abs(obs.y - centers_y).max() <= specs.JITTER_M + 1e-9
    assert obs.roll_deg.abs().max() <= 5 and obs.pitch_deg.abs().max() <= 5
    assert obs.contrast.between(0.85, 1.15).all() and obs.brightness.between(-0.15, 0.15).all()
    assert obs.blur_sigma.between(0, 1.5).all()
    assert abs(obs.yaw_deg.std() - 5.0) < 0.5
    assert sorted(obs.variant.unique()) == list(range(specs.N_OBS_VARIANTS))


def test_map_spec_applies_alignment_and_no_perturbation():
    map_rows = specs.map_spec(SITE, "pre", 100)
    grid = specs.site_grid(SITE)
    centers_x, centers_y = grid.center(map_rows.cell_row.to_numpy(), map_rows.cell_col.to_numpy())
    np.testing.assert_allclose(map_rows.x - centers_x, 2.5)
    np.testing.assert_allclose(map_rows.y - centers_y, -1.0)
    assert (map_rows[["yaw_deg", "roll_deg", "pitch_deg", "brightness", "blur_sigma", "noise_std"]] == 0).all().all()


def test_crop_ids_are_unique_across_specs():
    names = ["v3_fake_pre_map_60m", "v3_fake_post_obs_60m"]
    ids = [specs.with_crop_ids(specs.map_spec(SITE, "pre", 60), names[0]).crop_id,
           specs.with_crop_ids(specs.obs_spec(SITE, 60), names[1]).crop_id]
    assert not set(ids[0]).intersection(ids[1])


def test_canoas_grid_is_the_utm_box_and_reuses_poa_sources():
    site = specs.load_site("canoas")
    grid = specs.site_grid(site)
    assert (grid.x0, grid.y1, grid.nrows, grid.ncols) == (474000.0, 6695000.0, 320, 480)
    poa = specs.load_site("poa")
    assert site.sources == {d: poa.sources[d] for d in ("poa_post", "poa_pre_feb24")}
    assert site.offsets["poa_pre_feb24"] == poa.offsets["poa_pre_feb24"]
