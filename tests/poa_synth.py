"""Small synthetic Porto Alegre dataset in a temporary folder (no imagery, no network)."""
import json

import numpy as np
from scipy.ndimage import gaussian_filter

from dal.config import load_config


def synth_cfg(tmp):
    cfg = load_config("configs_poa/poa.yaml")
    cfg["paths"].update(data=str(tmp / "data_poa"), results=str(tmp / "results_poa"), runs=str(tmp / "runs_poa"),
                        report=str(tmp / "REPORT_POA.md"), taquari_data=str(tmp / "tq"))
    cfg["data"]["aoi"] = dict(lon0=-51.29, lon1=-51.25, lat0=-30.0, lat1=-29.97)
    cfg["regions"] = {"__replace__": True}
    cfg["regions"] = dict(ilhas=dict(lon0=-51.29, lon1=-51.255, lat0=-30.0, lat1=-29.97),
                          centro=dict(lon0=-51.285, lon1=-51.25, lat0=-30.0, lat1=-29.97))
    cfg["filter"]["window"] = 24
    cfg["filter"]["max_steps"] = 20
    cfg["eval"].update(min_windows=1, bootstrap=100, save_traj=2)
    return cfg


def make_dataset(cfg, seed=0):
    from dal.config import paths
    from dal.poa_data import poa_grid

    rng = np.random.default_rng(seed)
    g = poa_grid(cfg)
    R, C, D = g.nrows, g.ncols, 224
    P = paths(cfg)
    fd = P["data"] / "features"
    fd.mkdir(parents=True)
    base = gaussian_filter(rng.normal(size=(R, C, D)), sigma=(1, 1, 0))
    masks = {}
    for i, date in enumerate(cfg["data"]["sources"]):
        m = base + 0.3 * i * rng.normal(size=base.shape)
        np.save(fd / f"hand_{date}_map.npy", m.reshape(R * C, D).astype(np.float16))
        masks.update({f"{date}_valid_frac": np.ones((R, C), np.float32), f"{date}_cloud_frac": np.zeros((R, C), np.float32),
                      f"{date}_water_frac": rng.random((R, C)).astype(np.float32), f"{date}_ok": rng.random((R, C)) > 0.03})
    obs = np.stack([base + 0.2 * rng.normal(size=base.shape) for _ in range(2)]).reshape(2, R * C, D)
    np.save(fd / f"hand_{cfg['data']['obs_date']}_obs.npy", obs.astype(np.float16))
    np.savez_compressed(P["data"] / "masks.npz", **masks, grid=json.dumps(g.to_dict()))
    q, _ = np.linalg.qr(rng.normal(size=(D, 4)))
    for date in cfg["taquari_models"].values():
        np.savez(P["data"] / f"taquari_pca_hand_{date}.npz", mu=np.zeros(D, np.float32), Vt=q.T.astype(np.float32),
                 sd=np.full(4, 0.1, np.float32), reproduction_max_abs_err=np.array(0.0))
    tq = P["data"].parent / "tq"
    tq.mkdir()
    (tq / "beta.json").write_text(json.dumps({f"hand_{s}": dict(beta=20.0, beta1=20.0, kappa=1.0) for s in ("S0", "S1", "S2")}))
    return P
