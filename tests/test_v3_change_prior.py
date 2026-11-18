"""Prior de mudança: rasterização na grade e teste de proveniência (nunca lê a imagem pós-evento)."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from dal.v3.localization import change_prior


def _write_mask(path, mask, x0=1000.0, y1=5000.0, res=10.0):
    profile = dict(driver="GTiff", width=mask.shape[1], height=mask.shape[0], count=1, dtype="uint8",
                   crs="EPSG:32722", transform=Affine(res, 0, x0, 0, -res, y1))
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(mask.astype(np.uint8), 1)


def test_rasterized_prior_is_the_flooded_fraction_per_cell(tmp_path):
    mask = np.zeros((10, 10), np.uint8)
    mask[:, :5] = 1  # metade oeste inundada (50 m de 100 m)
    path = tmp_path / "taquari_prior_emsr720_10m.tif"
    _write_mask(path, mask)
    prior = change_prior.rasterize_prior(path, x0=1000.0, y1=5000.0, n_rows=4, n_cols=4, cell_m=25.0)
    np.testing.assert_allclose(prior[:, 0], 1.0)
    np.testing.assert_allclose(prior[:, 3], 0.0)
    assert ((prior >= 0) & (prior <= 1)).all()


def test_prior_code_refuses_the_post_event_image(tmp_path):
    for forbidden in sorted(change_prior.post_event_sources()):
        with pytest.raises(PermissionError):
            change_prior.assert_independent_source(forbidden if "://" in forbidden else tmp_path / forbidden)


def test_prior_rasterization_never_opens_post_event_files(tmp_path, monkeypatch):
    opened = []
    real_open = rasterio.open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", spy)
    path = tmp_path / "poa_prior_emsn194_10m.tif"
    _write_mask(path, np.ones((5, 5)))
    change_prior.rasterize_prior(path, x0=1000.0, y1=5000.0, n_rows=2, n_cols=2)
    forbidden = change_prior.post_event_sources()
    assert opened and not any(f in p for p in opened for f in forbidden)
