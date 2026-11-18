"""Ajuste fino contrastivo: InfoNCE, células só da faixa de treino (vazamento) e retomada."""
import numpy as np
import torch
import torch.nn as nn

from dal.v3.descriptors import finetune
from dal.v3.evaluation.folds import make_fold
from dal.v3.sensor import specs

SITE = specs.Site(name="fake", aoi={"lon0": -51.95, "lon1": -51.93, "lat0": -29.33, "lat1": -29.235},
                  obs_date="post", map_dates=("pre", "post"), sources={"pre": "a", "post": "b"}, offsets={})


class TinyBackbone(nn.Module):
    """Imita a interface do DINOv2: forward_features → tokens de patch."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.patch = nn.Conv2d(3, 8, 8, stride=8)

    def forward_features(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = torch.relu(block(tokens))
        return {"x_norm_patchtokens": tokens}


class FakeSampler:
    """Pares sintéticos em que o mapa e a observação compartilham um padrão por célula."""

    def __init__(self, n_cells=64, batch=16):
        self.patterns = torch.rand(n_cells, 3, 16, 16)
        self.batch = batch
        self.rng = np.random.default_rng(0)

    def sample(self):
        cells = self.rng.choice(len(self.patterns), self.batch, replace=False)
        base = self.patterns[cells]
        return base, base + 0.05 * torch.randn_like(base), cells


def test_info_nce_is_low_for_matched_pairs():
    embeddings = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    matched = finetune.info_nce(embeddings, embeddings, 0.07)
    shuffled = finetune.info_nce(embeddings, embeddings[torch.randperm(8)], 0.07)
    assert matched < shuffled


def test_training_cells_stay_in_the_fold_training_band():
    grid = specs.site_grid(SITE)
    cells = finetune.training_cells(SITE, "A")
    fold = make_fold("A", grid.nrows, specs.CELL_M, 128)
    assert set(np.unique(cells // grid.ncols)) <= set(fold.train_rows.tolist())
    assert not set(np.unique(cells // grid.ncols)) & set(fold.test_rows.tolist())


def test_finetune_learns_and_resumes(tmp_path):
    cfg = finetune.FinetuneConfig(steps=60, batch_size=16, checkpoint_minutes=0.0, trainable_blocks=1,
                                  projection_dim=8, head_learning_rate=1e-2, learning_rate=1e-2)
    summary = finetune.train(cfg, tmp_path, "cpu", backbone=TinyBackbone(), embed_dim=8, sampler=FakeSampler())
    assert summary["finished"] and summary["loss_last"] < summary["loss_first"]
    assert (tmp_path / "backbone.pt").exists()
    again = finetune.train(cfg, tmp_path, "cpu", backbone=TinyBackbone(), embed_dim=8, sampler=FakeSampler())
    assert again["step"] == 60
