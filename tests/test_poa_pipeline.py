"""Synthetic POA pipeline end to end: Taquari PCA applied without refitting, report sections,
and nothing created or modified in the protected Taquari folders."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from dal.config import ROOT
from poa_synth import make_dataset, synth_cfg

PROTECTED = [ROOT / "results", ROOT / "data", ROOT / "runs" / "overnight"]


def _snapshot():
    out = {}
    for d in PROTECTED:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_mtime_ns, st.st_size, hashlib.sha256(p.read_bytes()).hexdigest())
    return out


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from dal import poa, poa_report

    tmp = tmp_path_factory.mktemp("poa")
    before = _snapshot()
    cfg = synth_cfg(tmp)
    P = make_dataset(cfg)
    svd = np.linalg.svd

    def no_svd(*a, **k):
        raise AssertionError("PCA must not be refitted on POA data")

    np.linalg.svd = no_svd
    try:
        poa.build_cache(cfg)
    finally:
        np.linalg.svd = svd
    poa._MEM.clear()
    units = poa.eval_units(cfg)
    df = poa.evaluate(cfg, units, ["straight", "random", "greedy_map"], ["zeroshot"], 12, P["results"] / "episodes.csv",
                      1, 2)
    text = poa_report.build(cfg, P["results"], tmp / "REPORT_POA.md")
    return dict(cfg=cfg, P=P, df=df, text=text, units=units, before=before, after=_snapshot())


def test_protected_folders_untouched(pipeline):
    assert pipeline["before"] == pipeline["after"]
    assert (pipeline["P"]["results"] / "episodes.csv").exists()


def test_taquari_pca_applied_not_refitted(pipeline):
    from dal.poa import apply_pca, cache_dir, load_taquari_pca

    cfg, P = pipeline["cfg"], pipeline["P"]
    z = np.load(P["data"] / "taquari_pca_hand_pre_mar23.npz")
    loaded = load_taquari_pca(cfg, "pre_mar23")
    for k in ("mu", "Vt", "sd"):
        assert np.array_equal(loaded[k], z[k])
    m = np.load(cache_dir(cfg) / "map_poa_pre_feb24.npy").astype(np.float32)
    ok = np.load(P["data"] / "masks.npz")["poa_pre_feb24_ok"]
    stored = np.load(cache_dir(cfg) / "pca_S1_poa_pre_feb24.npy")
    assert np.allclose(stored, apply_pca(loaded, m, ok), atol=2e-3)  # cached map is float16


def test_episodes_paired_and_covariates(pipeline):
    df = pipeline["df"]
    assert len(pipeline["units"]) >= 4
    for _, g in df.groupby(["region", "scenario"]):
        w = g.groupby("method")[["window_r0", "window_c0"]].apply(lambda x: tuple(map(tuple, x.to_numpy())))
        assert len(set(w)) == 1  # same windows for every method
    ch = df.change_window.dropna()
    assert ch.between(0, 2).all()
    assert list(pipeline["P"]["results"].glob("traj/*.npz"))


def test_report_sections_and_columns(pipeline):
    from dal.poa_report import MAIN_COLS, SECTIONS

    text = pipeline["text"]
    for s in SECTIONS:
        assert s in text
    for h in ("### HP1", "### HP2", "### HP3"):
        assert h in text
    assert "| " + " | ".join(MAIN_COLS) + " |" in text
    s = pd.read_csv(pipeline["P"]["results"] / "summary.csv")
    assert {"success_lo", "success_hi", "false_loc", "distance_mean_m", "decision_ms"} <= set(s.columns)


@pytest.mark.data
def test_real_taquari_pca_file_matches_recomputation():
    from dal.config import load_config, paths
    from dal.poa import load_taquari_pca, taquari_pca_params

    cfg = load_config("configs_poa/poa.yaml")
    if not (paths(cfg)["data"] / "taquari_pca_hand_pre_mar23.npz").exists():
        pytest.skip("data_poa not prepared")
    for date in cfg["taquari_models"].values():
        a, b = load_taquari_pca(cfg, date), taquari_pca_params(cfg, date)
        for k in ("mu", "Vt", "sd"):
            assert np.array_equal(a[k], b[k])
