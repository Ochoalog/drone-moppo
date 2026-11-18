"""Extração de descritores (job do Colab) com modelo falso e fonte sintética: ordem, retomada e DONE.json."""
import json

import numpy as np
import pandas as pd

from dal.v3.descriptors import extract
from dal.v3.sensor.render import RasterGeoref


class FakeExtractor:
    """Descritor = média de cor do recorte (permite conferir a ordem das linhas)."""

    calls = 0

    def __call__(self, crops):
        FakeExtractor.calls += 1
        mean = crops.float().mean(dim=(1, 2)).numpy().astype(np.float16)
        return {"gem": mean, "cls": mean[:, :2]}


def _fake_source(cog_url, src_res, rows, margin_m):
    # Imagem cujo valor do canal 0 é a coluna e do canal 1 a linha: a média revela o centro do recorte.
    cols, rws = np.meshgrid(np.arange(256), np.arange(256))
    image = np.stack([cols, rws, np.full_like(cols, 7 if "a" in cog_url else 9)], -1).astype(np.uint8)
    return image, RasterGeoref(0.0, 256.0, 1.0)


def _spec(tmp_path, n=23):
    rng = np.random.default_rng(0)
    spec = pd.DataFrame({
        "crop_id": np.arange(n), "site": "t", "date_id": "d", "cog_url": np.where(np.arange(n) % 3, "a.tif", "b.tif"),
        "x": rng.uniform(60, 190, n), "y": rng.uniform(60, 190, n), "footprint_m": 20.0, "yaw_deg": 0.0,
        "roll_deg": 0.0, "pitch_deg": 0.0, "brightness": 0.0, "contrast": 1.0, "blur_sigma": 0.0, "noise_std": 0.0,
        "out_px": 16, "src_res_m": 1.0,
    })
    path = tmp_path / "spec.parquet"
    spec.to_parquet(path, index=False)
    return spec, path


def test_extraction_keeps_spec_order_resumes_and_writes_done(tmp_path, monkeypatch):
    spec, spec_path = _spec(tmp_path)
    monkeypatch.setattr(extract, "ROOT", tmp_path)
    config = {"spec": spec_path.name, "output_dir": "features/fake", "chunk_rows": 5, "batch_size": 4,
              "also_hand": True, "model": "fake"}
    metrics = extract.run(config, "job_x", tmp_path / "drive", tmp_path / "colab_out", "cpu",
                          extractor=FakeExtractor(), source_reader=_fake_source)
    gem = np.load(tmp_path / "drive/features/fake/gem.npy")
    assert gem.shape == (len(spec), 3)
    # A média da coluna/linha do recorte bate com o centro (x, y) de cada linha, na ordem da spec.
    np.testing.assert_allclose(gem[:, 0], spec.x - 0.5, atol=1.0)
    np.testing.assert_allclose(gem[:, 1], 256 - spec.y - 0.5, atol=1.0)
    assert np.load(tmp_path / "drive/features/fake/hand.npy").shape[0] == len(spec)
    assert (tmp_path / "drive/features/fake/MANIFEST.sha256").exists()
    done = json.loads((tmp_path / "colab_out/job_x/DONE.json").read_text())
    assert done["status"] == "OK" and done["metrics"]["n_rows"] == len(spec)

    # Retomada: com todos os blocos prontos, nada é renderizado de novo.
    calls_before = FakeExtractor.calls
    again = extract.run(config, "job_x", tmp_path / "drive", tmp_path / "colab_out", "cpu",
                        extractor=FakeExtractor(), source_reader=_fake_source)
    assert again["rendered_this_session"] == 0 and FakeExtractor.calls == calls_before
    assert metrics["rendered_this_session"] == len(spec)
