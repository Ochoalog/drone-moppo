"""Recuperação (top-1, top-5, top-1 a 1 célula) de descritores alinhados a uma spec: gate G1.

Usa **a mesma função do REPORT.md §2** (`dal.calibrate.window_stats`), com os mesmos parâmetros da v1
(20 janelas de treino 64×64, semente 0). Os descritores novos substituem só `map_desc` e `obs_desc`
do `ScenarioData` da v1, depois da mesma padronização (estatísticas das células de treino + L2).

Exemplos:
    # conferência: o hand da v1 pelo mesmo caminho precisa dar S1 = 0,409 e S2 = 0,269
    .venv/bin/python scripts/eval_retrieval_v3.py --descriptors hand-v1 --name hand_v1
    # DINOv2 zero-shot (arquivo float16 [N, D] na ordem da spec)
    .venv/bin/python scripts/eval_retrieval_v3.py --spec colab/specs/legacy_50m.parquet \
        --descriptors /caminho/legacy_50m.npy --name dinov2_vits14_gem
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dal.calibrate import window_stats
from dal.config import DATA, ROOT, load_config
from dal.env import ScenarioData
from dal.features import _standardize
from dal.grid import Grid
from dal.v3.reporting.provenance import with_provenance

V1_RETRIEVAL_WINDOWS = 20
V1_RETRIEVAL_SEED = 0
OUT = ROOT / "results_v3" / "retrieval"
MASKS_FILE = "colab/specs/legacy_50m_masks.npz"


def raw_arrays_from_spec(spec: pd.DataFrame, descriptors: np.ndarray, n_cells: int, map_date: str):
    """Monta (mapa (n, D), obs (K, n, D)) indexados por célula a partir das linhas da spec."""
    dim = descriptors.shape[1]
    map_rows = spec[(spec.kind == "map") & (spec.date_id == map_date)]
    map_raw = np.zeros((n_cells, dim), np.float32)
    # Lê do arquivo mapeado só as linhas usadas e converte para float32 (6144 dim não cabem inteiros na RAM).
    map_raw[map_rows.cell_index.to_numpy()] = np.asarray(descriptors[np.sort(map_rows.index.to_numpy())],
                                                         np.float32)[np.argsort(np.argsort(map_rows.index.to_numpy()))]
    obs_rows = spec[spec.kind == "obs"]
    variants = sorted(obs_rows.variant.unique())
    obs_raw = np.zeros((len(variants), n_cells, dim), np.float32)
    for k, variant in enumerate(variants):
        rows = obs_rows[obs_rows.variant == variant]
        order = rows.index.to_numpy()
        obs_raw[k, rows.cell_index.to_numpy()] = np.asarray(descriptors[np.sort(order)],
                                                            np.float32)[np.argsort(np.argsort(order))]
    return map_raw, obs_raw


def raw_arrays_hand_v1(map_date: str):
    """Descritores hand brutos gravados pela v1 (conferência do caminho)."""
    features = DATA / "features"
    return np.load(features / f"hand_{map_date}_map.npy"), np.load(features / "hand_post_obs.npy")


def retrieval_for_scenario(config: dict, scenario: str, map_raw: np.ndarray, obs_raw: np.ndarray,
                           split: str = "train") -> dict:
    """Padroniza como a v1 e roda `window_stats` no ScenarioData da v1 com os descritores trocados.

    `split="test"`: as janelas saem da faixa de teste da v1 (sul, que fica dentro da faixa de teste da dobra A
    da v3); usado no G1 do ajuste fino, que viu a faixa de treino. A padronização continua com o treino.
    """
    data_cfg = config["data"]
    grid = Grid.from_aoi(**data_cfg["aoi"], cell=data_cfg["cell_m"])
    train_rows, _ = grid.split_rows(data_cfg["split_lat"], data_cfg["split_margin_m"])
    train_idx = (train_rows[:, None] * grid.ncols + np.arange(grid.ncols)[None]).ravel()
    map_desc = _standardize(map_raw, train_idx).reshape(grid.nrows, grid.ncols, -1)
    obs_desc = _standardize(obs_raw, train_idx).reshape(obs_raw.shape[0], grid.nrows, grid.ncols, -1)
    # ScenarioData montado direto das máscaras da v1 versionadas (colab/specs/legacy_50m_masks.npz): não depende
    # do cache local, então roda também no Colab. Distintividade, PCA e β não entram na recuperação.
    masks = np.load(ROOT / MASKS_FILE)
    map_date = config["scenarios"][scenario]
    rows, cols = grid.nrows, grid.ncols
    _, test_rows = grid.split_rows(data_cfg["split_lat"], data_cfg["split_margin_m"])
    data = ScenarioData(name=f"v3_{scenario}", map_desc=map_desc.astype(np.float32),
                        obs_desc=obs_desc.astype(np.float32), map_ok=masks[f"{map_date}_ok"], obs_ok=masks["post_ok"],
                        distinct=np.zeros((rows, cols), np.float32), pca=np.zeros((1, rows, cols), np.float32),
                        train_rows=train_rows, test_rows=test_rows, beta=1.0)
    if split == "test":  # window_stats sorteia janelas de data.train_rows: aponta para a faixa de teste
        data.train_rows = test_rows
    return window_stats(data, int(config["filter"]["window"]), V1_RETRIEVAL_WINDOWS, V1_RETRIEVAL_SEED,
                        config["filter"]["min_window_valid"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--descriptors", required=True, help="arquivo .npy alinhado à spec, ou 'hand-v1'")
    parser.add_argument("--spec", default="colab/specs/legacy_50m.parquet")
    parser.add_argument("--name", required=True)
    parser.add_argument("--scenarios", nargs="*", default=["S0", "S1", "S2"])
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="faixa das janelas: treino (G1 zero-shot) ou teste (G1 do ajuste fino)")
    parser.add_argument("--out", default=str(OUT), help="pasta do CSV (no Colab: colab_out/<job_id>)")
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    grid = Grid.from_aoi(**config["data"]["aoi"], cell=config["data"]["cell_m"])
    n_cells = grid.nrows * grid.ncols
    if args.descriptors != "hand-v1":
        spec = pd.read_parquet(ROOT / args.spec)
        descriptors = np.load(args.descriptors, mmap_mode="r")
        if len(descriptors) != len(spec):
            raise ValueError(f"descritores ({len(descriptors)}) e spec ({len(spec)}) com tamanhos diferentes")
    rows = []
    for scenario in args.scenarios:
        map_date = config["scenarios"][scenario]
        if args.descriptors == "hand-v1":
            map_raw, obs_raw = raw_arrays_hand_v1(map_date)
        else:
            map_raw, obs_raw = raw_arrays_from_spec(spec, descriptors, n_cells, map_date)
        stats = retrieval_for_scenario(config, scenario, map_raw, obs_raw, args.split)
        rows.append({"descriptor": args.name, "scenario": scenario, "split": args.split, **stats})
        print(f"[retrieval] {args.name} {scenario}: top1 {stats['top1']:.3f} top5 {stats['top5']:.3f} "
              f"top1(<=1) {stats['top1_within1']:.3f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = with_provenance(pd.DataFrame(rows), source_file=str(args.descriptors), config=config,
                            protocol=f"window_stats v1: {V1_RETRIEVAL_WINDOWS} janelas de {args.split} 64x64, "
                                     "semente 0",
                            fold="A (treino norte)" if args.split == "train" else "A (teste sul)",
                            region="taquari")
    table.to_csv(out / f"{args.name}{'_test' if args.split == 'test' else ''}.csv", index=False)


if __name__ == "__main__":
    main()
