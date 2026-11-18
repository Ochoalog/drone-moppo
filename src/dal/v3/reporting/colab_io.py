"""Contrato de arquivos com a IA-2: DONE.json, fila de jobs e validação no lado da IA-1."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from dal.v3.reporting.provenance import config_hash, git_commit

JOB_COLUMNS = ["job_id", "notebook", "config_path", "depends_on", "est_gpu_min", "priority", "status"]
DONE_REQUIRED = ["job_id", "git_commit", "config_hash", "gpu_minutes", "started_at", "finished_at", "status", "metrics"]
VALID_STATUS = {"OK", "INVALID_GATE", "FAILED"}


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def load_job_config(path: str | Path) -> dict:
    """Lê o YAML de um job (o hash do conteúdo identifica a config no DONE.json)."""
    return yaml.safe_load(Path(path).read_text())


def write_done(out_dir: Path, *, job_id: str, config: dict, started_at: str, gpu_minutes: float,
               metrics: dict, status: str = "OK", reason: str = "", commit: str | None = None,
               extra: dict | None = None) -> Path:
    """Grava `DONE.json` no formato do contrato."""
    if status not in VALID_STATUS:
        raise ValueError(f"status inválido: {status}")
    payload = {"job_id": job_id, "git_commit": commit or git_commit(), "config_hash": config_hash(config),
               "gpu_minutes": round(float(gpu_minutes), 2), "started_at": started_at, "finished_at": now_iso(),
               "status": status, "reason": reason, "metrics": metrics, **(extra or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "DONE.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def validate_done(done_path: Path, jobs: pd.DataFrame, current_commit: str, repo_root: Path) -> list[str]:
    """Lista os problemas de um DONE.json (vazio = válido): campos, job conhecido, commit e hash da config."""
    problems = []
    done = json.loads(Path(done_path).read_text())
    missing = [k for k in DONE_REQUIRED if k not in done]
    if missing:
        return [f"campos ausentes: {missing}"]
    if done["status"] not in VALID_STATUS:
        problems.append(f"status inválido: {done['status']}")
    job = jobs[jobs.job_id == done["job_id"]]
    if job.empty:
        return problems + [f"job desconhecido: {done['job_id']}"]
    if done["git_commit"] != current_commit:
        problems.append(f"commit {done['git_commit']} ≠ CURRENT_COMMIT {current_commit}")
    expected_hash = config_hash(load_job_config(repo_root / job.iloc[0].config_path))
    if done["config_hash"] != expected_hash:
        problems.append(f"config_hash {done['config_hash']} ≠ {expected_hash}")
    return problems


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: Path, pattern: str = "*") -> Path:
    """MANIFEST.sha256 no formato do `sha256sum` para os arquivos da pasta."""
    lines = [f"{sha256_file(p)}  {p.name}" for p in sorted(directory.glob(pattern))
             if p.is_file() and p.name != "MANIFEST.sha256"]
    manifest = directory / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest
