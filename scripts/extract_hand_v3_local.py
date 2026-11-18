"""Descritor hand da v1 nas specs v3 (25 m, multialtitude), na CPU local, em vários processos.

Serve à ablação de percepção (PPO-CA com hand) e ao primeiro teste de ponta a ponta com dados reais (G3).
Lê os COGs pela rede (overview da resolução das specs). Saída: data_v3/features/v3/hand/<spec>/
(hand.npy, valid_frac.npy, cloud_frac.npy), no layout que `dal.v3.pipeline` lê (feature = "hand").

Uso: .venv/bin/python scripts/extract_hand_v3_local.py --site taquari --altitudes 100 --workers 8
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from dal.config import ROOT  # noqa: E402
from dal.v3.descriptors import extract  # noqa: E402
from dal.v3.sensor import specs  # noqa: E402

CHUNK_ROWS = 20000


class HandOnly:
    """Extrator só com o hand (o `extract.run` acrescenta a qualidade do recorte)."""

    def __call__(self, crops):
        return extract.hand_extractor(crops)


def download_source(site: specs.Site, date: str, raw_dir: Path) -> str:
    """Baixa uma vez o overview do COG (resolução das specs) cobrindo o local inteiro; DEFLATE, sem perda."""
    from dal.data import read_box, save_raster

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{site.name}_{date}_{specs.SOURCE_RES_M:.3f}m.tif"
    if not path.exists():
        grid = specs.site_grid(site)
        reach = specs.footprint_m(max(specs.ALTITUDES_M)) * 0.75 + 300.0
        bounds = (grid.x0 - reach, grid.y1 - grid.nrows * grid.cell - reach, grid.x0 + grid.ncols * grid.cell + reach,
                  grid.y1 + reach)
        started = time.time()
        raster = read_box(site.sources[date], bounds, factor=2)
        save_raster(raster, path, jpeg=False)
        print(f"[hand_v3] baixado {path.name}: {raster.img.shape} em {time.time() - started:.0f}s", flush=True)
        del raster
    return str(path)


def shard_job(args: tuple) -> dict:
    import torch

    torch.set_num_threads(1)
    spec_path, name, out_root, shard = args
    job = {"spec": str(spec_path), "output_dir": f"features/v3/hand/{name}", "chunk_rows": CHUNK_ROWS,
           "batch_size": 128, "margin_m": 200.0, "model": "hand", "read_per_chunk": True}
    return extract.run(job, f"hand_{name}", Path(out_root), Path(out_root) / "_colab_out", "cpu",
                       extractor=HandOnly(), shard=shard)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default="taquari")
    parser.add_argument("--altitudes", type=int, nargs="+", default=[100])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-root", default=str(ROOT / "data_v3"))
    args = parser.parse_args()
    site = specs.load_site(args.site)
    spec_dir = Path(args.out_root) / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    local_sources = {date: download_source(site, date, Path(args.out_root) / "raw") for date in
                     (*site.map_dates, site.obs_date)}
    for name, frame in specs.iter_specs(site):
        if int(name.rsplit("_", 1)[1].rstrip("m")) not in args.altitudes:
            continue
        path = spec_dir / f"{name}.parquet"
        if not path.exists():
            # Mesma geometria da spec; só a origem dos pixels muda (cópia local do mesmo overview do COG).
            frame.assign(cog_url=frame.date_id.map(local_sources)).to_parquet(path, index=False)
        shards = [(path, name, args.out_root, (i, args.workers)) for i in range(args.workers)]
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(shard_job, shards))
        names = extract.assemble(Path(args.out_root) / "features/v3/hand" / name, len(frame))
        extract.write_manifest(Path(args.out_root) / "features/v3/hand" / name, "*.npy")
        print(f"[hand_v3] {name}: {len(frame)} recortes, {names} ({(time.time() - started) / 60:.1f} min)", flush=True)
        del frame



if __name__ == "__main__":
    main()
