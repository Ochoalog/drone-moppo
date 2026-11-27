"""Benchmark P3-latency: mede o tempo por passo de decisão em CPU (e GPU se disponível) para:
1. Pré-processamento + DINOv2 ViT-S/14 (224x224) + grid4 GeM pooling + projeção
2. Atualização do histogram filter bayesiano (grade 128x128 = 16.384 células, passo de predição + verossimilhança)
3. Forward da política PPO convolucional (belief-space actor-critic)

Compara o tempo total por passo com o orçamento físico de voo (6,0 s por passo de 50 m a 10 m/s).
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_RES = ROOT / "results_v4"
OUT_RES.mkdir(parents=True, exist_ok=True)


def time_fn(fn, n_warmup: int = 5, n_runs: int = 30) -> dict:
    for _ in range(n_warmup):
        fn()
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms": round(float(arr.std()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "n_runs": n_runs,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running latency benchmark on {device} ({platform.processor() or platform.machine()}, threads={torch.get_num_threads()})")

    # 1. DINOv2 ViT-S/14 (cache local em ~/.cache/torch/hub)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(device)
    head = torch.nn.Sequential(
        torch.nn.Linear(16 * 384, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
    ).eval().to(device)

    crop = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8, device=device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def step_dino():
        x = crop.permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        feats = model.forward_features(x)["x_norm_patchtokens"]  # (1, 256, 384)
        cells = feats.view(1, 4, 4, 4, 4, 384).permute(0, 1, 3, 2, 4, 5).reshape(1, 16, 16, 384)
        pooled = cells.clamp(min=1e-6).pow(3.0).mean(dim=2).pow(1.0 / 3.0).reshape(1, -1)
        return F.normalize(head(pooled), dim=-1)

    # 2. Histogram filter update (128x128 grid, 256-d map descriptors)
    H, W, D = 128, 128, 256
    belief = torch.full((1, 1, H, W), 1.0 / (H * W), device=device)
    kernel = torch.ones((1, 1, 5, 5), device=device) / 25.0
    map_desc = F.normalize(torch.randn((H * W, D), device=device), dim=-1)
    obs_desc = F.normalize(torch.randn((D,), device=device), dim=-1)
    beta = 15.0

    @torch.inference_mode()
    def step_filter():
        # Motion shift + Gaussian blur prediction
        shifted = torch.roll(belief, shifts=(2, 0), dims=(2, 3))
        pred = F.conv2d(shifted, kernel, padding=2)
        # Observation likelihood
        sim = (map_desc @ obs_desc).view(1, 1, H, W)
        lik = torch.exp(beta * (sim - sim.max()))
        post = pred * lik
        return post / post.sum().clamp(min=1e-12)

    # 3. PPO belief-space CNN encoder + actor-critic head (arquitetura v3)
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((4, 4)),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 16, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).eval().to(device)
    obs_grid = torch.randn((1, 3, H, W), device=device)

    @torch.inference_mode()
    def step_ppo():
        return cnn(obs_grid)

    t_dino = time_fn(step_dino, n_warmup=3, n_runs=20)
    t_filt = time_fn(step_filter, n_warmup=5, n_runs=50)
    t_ppo = time_fn(step_ppo, n_warmup=5, n_runs=50)
    total_mean = round(t_dino["mean_ms"] + t_filt["mean_ms"] + t_ppo["mean_ms"], 2)
    total_p95 = round(t_dino["p95_ms"] + t_filt["p95_ms"] + t_ppo["p95_ms"], 2)

    res = {
        "device": device,
        "cpu_threads": torch.get_num_threads(),
        "grid": f"{H}x{W}",
        "descriptor_dim": D,
        "dinov2_vits14_grid4_ms": t_dino,
        "histogram_filter_ms": t_filt,
        "ppo_policy_ms": t_ppo,
        "total_step_mean_ms": total_mean,
        "total_step_p95_ms": total_p95,
        "step_budget_s": 6.0,
        "budget_utilization_pct": round(100.0 * (total_mean / 6000.0), 2),
    }
    (OUT_RES / "p3_latency_cpu.json").write_text(json.dumps(res, indent=2))
    print("\n=== P3 LATENCY BENCHMARK ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
