"""Extração de descritores a partir de uma spec (job da IA-2 no Colab; também roda na CPU).

Para cada fonte (`cog_url`, `src_res_m`) da spec: lê o raster uma vez (janela que cobre os recortes,
no overview da resolução pedida), renderiza os recortes com `dal.v3.sensor.render` e extrai os
descritores pedidos (DINOv2 com GeM e CLS; opcionalmente o hand da v1 sobre os mesmos pixels).

Retomável: grava blocos (`chunk_XXXXX.npz`) na pasta de saída e pula os que já existem. No fim, junta
tudo em `<descritor>.npy` (float16 [N, D], na ordem da spec), grava `MANIFEST.sha256`, um resumo
pequeno e o `DONE.json` em `colab_out/<job_id>/`.

Uso (Colab): python -m dal.v3.descriptors.extract --job-config configs_v3/jobs/g1_legacy_dinov2.yaml \
    --job-id g1_legacy_dinov2 --out-root /content/drive/MyDrive/drone_v3 --colab-out colab_out
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dal.config import ROOT
from dal.v3.reporting.colab_io import load_job_config, now_iso, write_done, write_manifest
from dal.v3.sensor.render import SOURCE_RES_COLUMN, RasterGeoref, batch_from_spec, render_crops

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GEM_P = 3.0


# ============================================================================ modelos
def preprocess_for_dino(crops: torch.Tensor, input_px: int) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, input_px, input_px) normalizado pela média e desvio do ImageNet."""
    x = crops.permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(input_px, input_px), mode="bilinear", align_corners=False,
                                        antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def grid_gem(patches: torch.Tensor, grid: int) -> torch.Tensor:
    """GeM em uma grade grid × grid dos tokens de patch (preserva o arranjo espacial dentro do recorte)."""
    n, tokens, dim = patches.shape
    side = int(round(tokens ** 0.5))
    cells = patches.view(n, grid, side // grid, grid, side // grid, dim).permute(0, 1, 3, 2, 4, 5)
    cells = cells.reshape(n, grid * grid, -1, dim)
    return cells.clamp(min=1e-6).pow(GEM_P).mean(dim=2).pow(1.0 / GEM_P).reshape(n, grid * grid * dim)


class DinoV2Extractor:
    """DINOv2 (torch.hub) com GeM sobre os tokens de patch e o token CLS; entrada redimensionada para input_px."""

    def __init__(self, model_name: str, input_px: int, device: str, model=None, weights: str | None = None,
                 grids: tuple[int, ...] = ()):
        self.model = model if model is not None else torch.hub.load("facebookresearch/dinov2", model_name)
        if weights:
            # Pesos do ajuste fino contrastivo (só o backbone; a cabeça de projeção não entra no descritor).
            state = torch.load(weights, map_location="cpu")
            self.model.load_state_dict(state.get("backbone", state), strict=False)
        self.model = self.model.eval().to(device)
        self.input_px, self.device = input_px, device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.use_half = device.startswith("cuda")
        self.grids = tuple(grids)  # saídas extras "grid<k>": GeM espacial k × k (opcional)

    @torch.inference_mode()
    def __call__(self, crops: torch.Tensor) -> dict[str, np.ndarray]:
        x = preprocess_for_dino(crops.to(self.device), self.input_px)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_half):
            features = self.model.forward_features(x)
        patches = features["x_norm_patchtokens"].float()
        out = {"gem": patches.clamp(min=1e-6).pow(GEM_P).mean(dim=1).pow(1.0 / GEM_P),
               "cls": features["x_norm_clstoken"].float()}
        for grid in self.grids:
            out[f"grid{grid}"] = grid_gem(patches, grid)
        return {name: value.cpu().numpy().astype(np.float16) for name, value in out.items()}


# Detector de nuvem da v1 (claro e pouco saturado), sem a morfologia: só a fração por recorte.
CLOUD_V_MIN, CLOUD_S_MAX = 200, 40


def crop_quality(crops: torch.Tensor) -> dict[str, np.ndarray]:
    """Fração de pixels válidos (dado ≠ 0) e de pixels de nuvem por recorte, para as máscaras na grade v3."""
    x = crops.float()
    valid = (x.amax(dim=-1) > 3).float()
    value = x.amax(dim=-1)
    saturation = torch.where(value > 0, (value - x.amin(dim=-1)) / value.clamp(min=1) * 255, torch.zeros_like(value))
    cloud = ((value > CLOUD_V_MIN) & (saturation < CLOUD_S_MAX)).float()
    return {"valid_frac": valid.mean(dim=(1, 2)).cpu().numpy()[:, None].astype(np.float16),
            "cloud_frac": cloud.mean(dim=(1, 2)).cpu().numpy()[:, None].astype(np.float16)}


def hand_extractor(crops: torch.Tensor) -> dict[str, np.ndarray]:
    """Descritor hand da v1 sobre os mesmos pixels (comparação justa no G1)."""
    from dal.features import hand_descriptor

    return {"hand": hand_descriptor(crops.cpu().numpy()).astype(np.float16)}


# ============================================================================ leitura das fontes
def read_source(cog_url: str, src_res_m: float, rows: pd.DataFrame, margin_m: float):
    """Lê a janela da fonte que cobre todos os recortes das linhas, no overview mais próximo de src_res_m."""
    import rasterio

    from dal.data import read_box

    path = cog_url if cog_url.startswith(("/", "/vsi")) else "/vsicurl/" + cog_url
    with rasterio.open(path) as dataset:
        native_res = dataset.transform.a
    factor = max(1, int(round(src_res_m / native_res)))
    if factor > 1 and 2 ** int(round(math.log2(factor))) != factor:
        raise ValueError(f"fator de overview {factor} não é potência de 2")
    reach = float(rows.footprint_m.max()) * 0.75 + margin_m  # meia diagonal do footprint + folga
    bounds = (rows.x.min() - reach, rows.y.min() - reach, rows.x.max() + reach, rows.y.max() + reach)
    raster = read_box(cog_url, bounds, factor)
    return raster.img, RasterGeoref(raster.x_origin, raster.y_origin, raster.res)


# ============================================================================ blocos retomáveis
def chunk_path(work_dir: Path, index: int) -> Path:
    return work_dir / f"chunk_{index:05d}.npz"


def run(config: dict, job_id: str, out_root: Path, colab_out: Path, device: str, extractor=None,
        source_reader=read_source) -> dict:
    """Executa (ou retoma) a extração; devolve as métricas gravadas no DONE.json."""
    started_at, started = now_iso(), time.time()
    spec = pd.read_parquet(Path(config["spec"]) if Path(config["spec"]).is_absolute() else ROOT / config["spec"])
    work_dir = out_root / config["output_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows = int(config.get("chunk_rows", 20000))
    extractor = extractor or DinoV2Extractor(config["model"], int(config.get("input_px", 224)), device,
                                             weights=config.get("weights"))
    also_hand = bool(config.get("also_hand", False))
    margin_m = float(config.get("margin_m", 50.0))
    batch_size = int(config.get("batch_size", 256))

    rendered = 0
    spec = spec.assign(_order=np.arange(len(spec)))
    for (cog_url, src_res), source_rows in spec.groupby(["cog_url", SOURCE_RES_COLUMN], sort=False):
        chunk_ids = source_rows._order.to_numpy() // chunk_rows
        key = _source_key(cog_url, src_res)
        pending = [c for c in np.unique(chunk_ids) if not chunk_path(work_dir, int(c) * 1000 + key).exists()]
        if not pending:
            continue
        image, georef = source_reader(cog_url, float(src_res), source_rows, margin_m)
        image_t = torch.as_tensor(image, device=device)
        for chunk in pending:
            rows = source_rows[chunk_ids == chunk]
            outputs: dict[str, list] = {}
            for start in range(0, len(rows), batch_size):
                batch_rows = rows.iloc[start:start + batch_size]
                crops = render_crops(image_t, georef, batch_from_spec(batch_rows, device))
                parts = extractor(crops) | crop_quality(crops)
                if also_hand:
                    parts |= hand_extractor(crops)
                for name, values in parts.items():
                    outputs.setdefault(name, []).append(values)
            arrays = {name: np.concatenate(v) for name, v in outputs.items()}
            np.savez(chunk_path(work_dir, int(chunk) * 1000 + key),
                     order=rows._order.to_numpy(), **arrays)
            rendered += len(rows)
        del image_t, image

    names = assemble(work_dir, len(spec))
    write_manifest(work_dir, "*.npy")
    minutes = (time.time() - started) / 60
    metrics = {"n_rows": len(spec), "rendered_this_session": rendered, "descriptors": names,
               "crops_per_second": rendered / max(time.time() - started, 1e-9), "device": device,
               "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    out_dir = colab_out / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**metrics, "descriptors": ",".join(names)}]).to_csv(out_dir / "summary.csv", index=False)
    write_done(out_dir, job_id=job_id, config=config, started_at=started_at,
               gpu_minutes=minutes if device.startswith("cuda") else 0.0, metrics=metrics,
               extra={"artifacts": str(work_dir)})
    return metrics


def _source_key(cog_url: str, src_res: float) -> int:
    """Número estável (0–999) por fonte, para os blocos de fontes diferentes não colidirem."""
    import zlib

    return zlib.crc32(f"{cog_url}|{src_res}".encode()) % 1000


def assemble(work_dir: Path, n_rows: int) -> list[str]:
    """Junta os blocos em um .npy por descritor (float16 [N, D]) na ordem da spec."""
    chunks = sorted(work_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"nenhum bloco em {work_dir}")
    first = np.load(chunks[0])
    names = [k for k in first.files if k != "order"]
    for name in names:
        full = np.lib.format.open_memmap(work_dir / f"{name}.npy", mode="w+", dtype=np.float16,
                                         shape=(n_rows, first[name].shape[1]))
        filled = np.zeros(n_rows, bool)
        for path in chunks:
            part = np.load(path)
            full[part["order"]] = part[name]
            filled[part["order"]] = True
        full.flush()
        if not filled.all():
            raise RuntimeError(f"{name}: {int((~filled).sum())} linhas sem descritor")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out-root", required=True, help="raiz dos artefatos grandes (Drive: MyDrive/drone_v3)")
    parser.add_argument("--colab-out", default="colab_out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_job_config(ROOT / args.job_config)
    print(run(config, args.job_id, Path(args.out_root), Path(args.colab_out), args.device))


if __name__ == "__main__":
    main()
