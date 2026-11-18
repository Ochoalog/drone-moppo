"""Colunas de proveniência obrigatórias em todo CSV da v3 (§1, princípio 4, do plano v3)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from dal.config import ROOT

PROVENANCE_COLUMNS = ["source_file", "protocol", "n_episodes", "seeds", "fold", "region", "git_commit", "config_hash"]


def git_commit(repo: Path = ROOT) -> str:
    """Commit atual; com mudanças não commitadas, acrescenta o sufixo `-dirty`."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, cwd=repo).stdout.strip()
    return f"{head}-dirty" if dirty else head


def config_hash(config: dict) -> str:
    """Hash curto e estável de uma config (JSON com chaves ordenadas)."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def with_provenance(frame: pd.DataFrame, *, source_file: str, protocol: str, config: dict,
                    n_episodes: int | None = None, seeds: str = "", fold: str = "", region: str = "") -> pd.DataFrame:
    """Acrescenta as colunas de proveniência (sem sobrescrever as que o frame já tiver por linha)."""
    values = {"source_file": source_file, "protocol": protocol, "n_episodes": n_episodes, "seeds": seeds,
              "fold": fold, "region": region, "git_commit": git_commit(), "config_hash": config_hash(config)}
    frame = frame.copy()
    for column, value in values.items():
        if column not in frame.columns:
            frame[column] = value
    return frame
