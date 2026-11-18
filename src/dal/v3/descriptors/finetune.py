"""Ajuste fino contrastivo do DINOv2 (notebook 03; §3.1 do plano v3).

Pares positivos: recorte do **mapa** (data pré-evento, sem perturbação) e recorte de uma **variante de
observação** pós-evento da mesma célula (com deslocamento, yaw, inclinação, fotometria), na mesma altitude.
Negativos: as demais células do lote. Perda InfoNCE simétrica sobre GeM → projeção → L2.

Sem vazamento: as células vêm **só** da faixa de treino da dobra (o teste confere), e só do Taquari.
Retomável: checkpoint no Drive a cada ≤ `checkpoint_minutes`; ao recomeçar, continua do último passo.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional

from dal.config import ROOT
from dal.v3.descriptors.extract import GEM_P, IMAGENET_MEAN, IMAGENET_STD, read_source
from dal.v3.evaluation.folds import make_fold
from dal.v3.reporting.colab_io import load_job_config, now_iso, write_done
from dal.v3.sensor import specs
from dal.v3.sensor.render import batch_from_spec, render_crops


@dataclass
class FinetuneConfig:
    site: str = "taquari"
    fold: str = "A"
    map_dates: tuple[str, ...] = ("pre_mar23", "pre_oct23")
    altitudes: tuple[int, ...] = specs.ALTITUDES_M
    model: str = "dinov2_vits14"
    trainable_blocks: int = 4
    projection_dim: int = 256
    temperature: float = 0.07
    batch_size: int = 96
    steps: int = 6000
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    input_px: int = 224
    checkpoint_minutes: float = 15.0
    max_minutes: float = 170.0  # jobs ≤ 3 h no Colab
    seed: int = 0


class ContrastiveModel(nn.Module):
    """Backbone (DINOv2) + GeM + projeção. Só os últimos `trainable_blocks` blocos treinam."""

    def __init__(self, backbone: nn.Module, embed_dim: int, projection_dim: int, trainable_blocks: int):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        blocks = getattr(self.backbone, "blocks", [])
        for block in list(blocks)[-trainable_blocks:] if trainable_blocks else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, projection_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.backbone.forward_features(images)["x_norm_patchtokens"]
        gem = patches.clamp(min=1e-6).pow(GEM_P).mean(dim=1).pow(1.0 / GEM_P)
        return functional.normalize(self.head(gem), dim=-1)


def info_nce(map_embeddings: torch.Tensor, obs_embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE simétrico: a i-ésima observação casa com o i-ésimo mapa; os demais do lote são negativos."""
    logits = obs_embeddings @ map_embeddings.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return 0.5 * (functional.cross_entropy(logits, targets) + functional.cross_entropy(logits.T, targets))


def training_cells(site: specs.Site, fold: str) -> np.ndarray:
    """Índices das células da faixa de treino da dobra (o único conjunto que o ajuste fino pode ver)."""
    grid = specs.site_grid(site)
    rows = make_fold(fold, grid.nrows, specs.CELL_M, 128).train_rows
    return (rows[:, None] * grid.ncols + np.arange(grid.ncols)[None]).ravel()


def preprocess(crops: torch.Tensor, input_px: int) -> torch.Tensor:
    x = crops.permute(0, 3, 1, 2).float() / 255.0
    x = functional.interpolate(x, size=(input_px, input_px), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class PairSampler:
    """Sorteia lotes de pares (mapa, observação) de células de treino e os renderiza na GPU."""

    def __init__(self, site: specs.Site, cfg: FinetuneConfig, images: dict, device: str):
        self.site, self.cfg, self.images, self.device = site, cfg, images, device
        self.cells = training_cells(site, cfg.fold)
        self.rng = np.random.default_rng(cfg.seed)
        self._specs: dict = {}

    def _spec(self, kind: str, date: str, altitude: int) -> pd.DataFrame:
        """Specs geradas uma vez por (tipo, data, altitude) e reaproveitadas em todos os lotes."""
        key = (kind, date, altitude)
        if key not in self._specs:
            self._specs[key] = (specs.map_spec(self.site, date, altitude) if kind == "map"
                                else specs.obs_spec(self.site, altitude))
        return self._specs[key]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        cfg = self.cfg
        cells = self.rng.choice(self.cells, cfg.batch_size, replace=False)
        altitude = int(self.rng.choice(cfg.altitudes))
        map_date = str(self.rng.choice(cfg.map_dates))
        variant = int(self.rng.integers(specs.N_OBS_VARIANTS))
        map_rows = self._spec("map", map_date, altitude).iloc[cells]
        obs_all = self._spec("obs", self.site.obs_date, altitude)
        n_cells = len(obs_all) // specs.N_OBS_VARIANTS
        obs_rows = obs_all.iloc[variant * n_cells + cells]
        pair = []
        for rows, date in ((map_rows, map_date), (obs_rows, self.site.obs_date)):
            image, georef = self.images[date]
            rows = rows.assign(crop_id=np.arange(len(rows)) + int(self.rng.integers(1 << 30)))
            pair.append(preprocess(render_crops(image, georef, batch_from_spec(rows, self.device)), cfg.input_px))
        return pair[0], pair[1], cells


def load_images(site: specs.Site, cfg: FinetuneConfig, device: str) -> dict:
    """Lê cada data uma vez (resolução das specs v3) e mantém na GPU."""
    images = {}
    reference = specs.map_spec(site, site.obs_date, max(cfg.altitudes))
    for date in (*cfg.map_dates, site.obs_date):
        image, georef = read_source(site.sources[date], specs.SOURCE_RES_M, reference, 200.0)
        images[date] = (torch.as_tensor(image, device=device), georef)
    return images


def train(cfg: FinetuneConfig, out_dir: Path, device: str, backbone: nn.Module | None = None,
          embed_dim: int = 384, sampler=None, log=print) -> dict:
    """Treina (ou retoma); grava `checkpoint.pt` e, no fim, `backbone.pt` (pesos usados pelo extrator)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    backbone = backbone if backbone is not None else torch.hub.load("facebookresearch/dinov2", cfg.model)
    model = ContrastiveModel(backbone, embed_dim, cfg.projection_dim, cfg.trainable_blocks).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg.learning_rate},
        {"params": model.head.parameters(), "lr": cfg.head_learning_rate}], weight_decay=0.01)
    step, history = 0, []
    checkpoint = out_dir / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, history = saved["step"], saved["history"]
        log(f"[finetune] retomando do passo {step}")
    if sampler is None:
        site = specs.load_site(cfg.site)
        sampler = PairSampler(site, cfg, load_images(site, cfg, device), device)
    sampler.rng = np.random.default_rng([cfg.seed, step])  # retomada: continua com sorteios novos
    started = last_save = time.time()
    use_amp = device.startswith("cuda")
    while step < cfg.steps and (time.time() - started) / 60 < cfg.max_minutes:
        map_images, obs_images, _ = sampler.sample()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = info_nce(model(map_images), model(obs_images), cfg.temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        history.append(loss.item())
        if time.time() - last_save >= cfg.checkpoint_minutes * 60:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "history": history}, checkpoint)
            last_save = time.time()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "history": history}, checkpoint)
    finished = step >= cfg.steps
    if finished:
        torch.save({"backbone": model.backbone.state_dict()}, out_dir / "backbone.pt")
    window = max(1, min(100, len(history) // 5))  # janela das médias: até 100 passos, no máximo 1/5 do total
    summary = {"step": step, "finished": finished, "loss_window": window,
               "loss_first": float(np.mean(history[:window])) if history else None,
               "loss_last": float(np.mean(history[-window:])) if history else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--colab-out", default="colab_out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    raw = load_job_config(ROOT / args.job_config)
    cfg = FinetuneConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    started_at, started = now_iso(), time.time()
    summary = train(cfg, Path(args.out_root) / "finetune" / args.job_id, args.device)
    write_done(Path(args.colab_out) / args.job_id, job_id=args.job_id, config=raw, started_at=started_at,
               gpu_minutes=(time.time() - started) / 60, metrics=summary,
               status="OK" if summary["finished"] else "FAILED",
               reason="" if summary["finished"] else "orçamento de tempo do job esgotado; rode de novo para retomar")
    pd.DataFrame([summary]).to_csv(Path(args.colab_out) / args.job_id / "summary.csv", index=False)


if __name__ == "__main__":
    main()
