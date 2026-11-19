"""Specs dos recortes v3 (§3.3–3.4 do plano): grade de 25 m, 3 altitudes, mapa sem perturbação e K = 4
variantes de observação por célula e altitude.

Determinístico dada a semente: a IA-1 commita só o manifesto (`colab/specs/v3_manifest.json`, com o hash
do **conteúdo** de cada spec) e a IA-2 gera as specs no Colab com este mesmo código, conferindo os hashes
(`--verify`). O hash é do conteúdo, não dos bytes do parquet, que mudam com a versão do pyarrow.

Modelo do sensor (variantes de observação):
- footprint = 2·h·tan(42°): 108 m a 60 m, 180 m a 100 m, 270 m a 150 m; saída de 224 px;
- deslocamento uniforme dentro da célula (±12,5 m em x e em y);
- yaw ~ N(0, 5°); roll e pitch ~ U(−5°, 5°);
- contraste ~ U(0,85; 1,15); brilho ~ U(−0,15; 0,15) (× 128 níveis); desfoque σ ~ U(0; 1,5) px; ruído σ = 3 níveis.

Uso:
    python -m dal.v3.sensor.specs --site taquari --out colab/specs            # gera e grava o manifesto
    python -m dal.v3.sensor.specs --site taquari --out /content/specs --verify  # Colab: gera e confere
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dal.config import ROOT, load_config
from dal.grid import Grid
from dal.v3.sensor.render import SOURCE_RES_COLUMN

CELL_M = 25.0
ALTITUDES_M = (60, 100, 150)
HALF_FOV_DEG = 42.0
OUT_PX = 224
SOURCE_RES_M = 0.6103515625  # overview 2× do COG de 0,305 m/px
N_OBS_VARIANTS = 4
SPEC_SEED = 20261118
JITTER_M = CELL_M / 2
YAW_SD_DEG = 5.0
TILT_MAX_DEG = 5.0
CONTRAST_RANGE = (0.85, 1.15)
BRIGHTNESS_RANGE = (-0.15, 0.15)
BLUR_SIGMA_RANGE = (0.0, 1.5)
NOISE_STD = 3.0
MANIFEST = "v3_manifest.json"
SITES = ("taquari", "poa", "canoas")
CANOAS_AOI_UTM = {"x0": 474000.0, "x1": 486000.0, "y0": 6687000.0, "y1": 6695000.0}  # EPSG:32722


@dataclass(frozen=True)
class Site:
    """Um local: AOI, data observada (pós-evento), datas de mapa e deslocamentos de alinhamento."""

    name: str
    aoi: dict
    obs_date: str
    map_dates: tuple[str, ...]
    sources: dict
    offsets: dict


def footprint_m(altitude_m: float) -> float:
    return 2 * altitude_m * math.tan(math.radians(HALF_FOV_DEG))


def load_site(name: str) -> Site:
    """Taquari e Porto Alegre a partir das configs da v1 (só leitura); Canoas com a AOI da V27."""
    # Deslocamentos copiados de data*/align.json (fora do git) para um arquivo versionado: o Colab precisa deles.
    raw = json.loads((ROOT / "configs_v3" / "alignment_offsets.json").read_text())
    if name in raw:
        offsets = {k: ((v["dx_m"], v["dy_m"]) if v["applied"] else (0.0, 0.0)) for k, v in raw[name].items()}
    if name == "taquari":
        config = load_config("configs/default.yaml")
        return Site("taquari", config["data"]["aoi"], "post", ("pre_mar23", "pre_oct23", "post"),
                    config["data"]["sources"], offsets)
    if name == "poa":
        config = load_config("configs_poa/poa.yaml")
        return Site("poa", config["data"]["aoi"], "poa_post", ("poa_pre_feb24", "poa_pre_dec22", "poa_post"),
                    config["data"]["sources"], offsets)
    if name == "canoas":
        # Terceira área (V27): mesmos COGs de Porto Alegre (pós 09/05/2024, pré P1 02/02/2024) e os mesmos
        # deslocamentos de alinhamento; AOI dada direto em UTM 22S (Mathias Velho e arredores, 96 km²).
        config = load_config("configs_poa/poa.yaml")
        sources = {date: config["data"]["sources"][date] for date in ("poa_post", "poa_pre_feb24")}
        offsets = {k: ((v["dx_m"], v["dy_m"]) if v["applied"] else (0.0, 0.0)) for k, v in raw["poa"].items()}
        return Site("canoas", CANOAS_AOI_UTM, "poa_post", ("poa_pre_feb24", "poa_post"), sources, offsets)
    raise ValueError(f"local desconhecido: {name}")


def site_grid(site: Site) -> Grid:
    if "x0" in site.aoi:  # AOI já em UTM (Canoas): a grade é a própria caixa
        aoi = site.aoi
        return Grid(float(aoi["x0"]), float(aoi["y1"]), int(round((aoi["y1"] - aoi["y0"]) / CELL_M)),
                    int(round((aoi["x1"] - aoi["x0"]) / CELL_M)), CELL_M)
    return Grid.from_aoi(**site.aoi, cell=CELL_M)


def _base_frame(site: Site, grid: Grid, date: str, altitude: float, n: int) -> pd.DataFrame:
    rows, cols = np.divmod(np.arange(n), grid.ncols)
    return pd.DataFrame({"site": site.name, "date_id": date, "cog_url": site.sources[date],
                         "footprint_m": footprint_m(altitude), "out_px": OUT_PX, SOURCE_RES_COLUMN: SOURCE_RES_M,
                         "altitude_m": int(altitude), "cell_index": np.arange(n), "cell_row": rows, "cell_col": cols})


def map_spec(site: Site, date: str, altitude: float) -> pd.DataFrame:
    """Mapa: centro da célula (com o alinhamento da data), sem perturbação."""
    grid = site_grid(site)
    n = grid.nrows * grid.ncols
    x, y = grid.center(*np.divmod(np.arange(n), grid.ncols))
    dx, dy = site.offsets.get(date, (0.0, 0.0))
    frame = _base_frame(site, grid, date, altitude, n)
    frame = frame.assign(kind="map", variant=-1, x=x + dx, y=y + dy, yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0,
                         brightness=0.0, contrast=1.0, blur_sigma=0.0, noise_std=0.0)
    return frame


def obs_spec(site: Site, altitude: float, n_variants: int = N_OBS_VARIANTS) -> pd.DataFrame:
    """K variantes de observação por célula, determinísticas dada (semente, local, altitude, variante)."""
    grid = site_grid(site)
    n = grid.nrows * grid.ncols
    x, y = grid.center(*np.divmod(np.arange(n), grid.ncols))
    dx, dy = site.offsets.get(site.obs_date, (0.0, 0.0))
    frames = []
    for k in range(n_variants):
        rng = np.random.default_rng([SPEC_SEED, zlib.crc32(site.name.encode()), int(altitude), k])
        frame = _base_frame(site, grid, site.obs_date, altitude, n)
        frames.append(frame.assign(
            kind="obs", variant=k,
            x=x + dx + rng.uniform(-JITTER_M, JITTER_M, n), y=y + dy + rng.uniform(-JITTER_M, JITTER_M, n),
            yaw_deg=rng.normal(0.0, YAW_SD_DEG, n), roll_deg=rng.uniform(-TILT_MAX_DEG, TILT_MAX_DEG, n),
            pitch_deg=rng.uniform(-TILT_MAX_DEG, TILT_MAX_DEG, n), brightness=rng.uniform(*BRIGHTNESS_RANGE, n),
            contrast=rng.uniform(*CONTRAST_RANGE, n), blur_sigma=rng.uniform(*BLUR_SIGMA_RANGE, n),
            noise_std=NOISE_STD))
    return pd.concat(frames, ignore_index=True)


def with_crop_ids(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """crop_id único entre specs: 20 bits altos do hash do nome + índice da linha."""
    prefix = zlib.crc32(name.encode()) & 0xFFFFF
    return frame.assign(crop_id=(np.int64(prefix) << 32) + np.arange(len(frame), dtype=np.int64))


def content_hash(frame: pd.DataFrame) -> str:
    """Hash do conteúdo (independe da versão do pyarrow e da ordem das colunas)."""
    ordered = frame[sorted(frame.columns)]
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return f"{zlib.crc32(values.tobytes()):08x}-{len(frame)}"


def iter_specs(site: Site):
    """(nome, DataFrame) de todas as specs de um local."""
    for altitude in ALTITUDES_M:
        for date in site.map_dates:
            name = f"v3_{site.name}_{date}_map_{altitude}m"
            yield name, with_crop_ids(map_spec(site, date, altitude), name)
        name = f"v3_{site.name}_{site.obs_date}_obs_{altitude}m"
        yield name, with_crop_ids(obs_spec(site, altitude), name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, choices=list(SITES))
    parser.add_argument("--out", default="colab/specs")
    parser.add_argument("--verify", action="store_true", help="confere os hashes contra o manifesto commitado")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "colab" / "specs" / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    site = load_site(args.site)
    for name, frame in iter_specs(site):
        digest = content_hash(frame)
        if args.verify:
            if manifest.get(name) != digest:
                raise SystemExit(f"{name}: hash {digest} ≠ manifesto {manifest.get(name)}")
        else:
            manifest[name] = digest
        frame.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(frame)} linhas, hash {digest}", flush=True)
    if not args.verify:
        manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))


if __name__ == "__main__":
    main()
