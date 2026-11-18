"""Publica o resultado pequeno de um job na branch `colab/results` (roda no Colab, pela IA-2).

Copia `colab_out/<job_id>/` (≤ 5 MB) para um worktree da branch `colab/results`, acrescenta uma linha
em `colab_out/LOG.md`, faz commit e push. Não toca na branch do código.

Uso: python scripts/colab_publish_v3.py --job-id g1_legacy_dinov2 --main-number "top1 S2 = 0.31" \
        --next-job g1_extract_v3_25m
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from dal.config import ROOT

RESULTS_BRANCH = "colab/results"
MAX_BYTES = 5 * 1024 * 1024


def run_git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def ensure_worktree(path: Path) -> Path:
    """Worktree da branch de resultados (cria a partir do remoto, se ainda não existir)."""
    if not path.exists():
        run_git("fetch", "origin", RESULTS_BRANCH, cwd=ROOT)
        run_git("worktree", "add", str(path), "-B", RESULTS_BRANCH, f"origin/{RESULTS_BRANCH}", cwd=ROOT)
    else:
        run_git("pull", "--ff-only", "origin", RESULTS_BRANCH, cwd=path)
    return path


def publish(job_id: str, source: Path, worktree: Path, main_number: str, next_job: str) -> None:
    size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    if size > MAX_BYTES:
        raise ValueError(f"{source} tem {size / 1e6:.1f} MB (> 5 MB): artefatos grandes vão para o Drive")
    done = json.loads((source / "DONE.json").read_text())
    target = worktree / "colab_out" / job_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    log = worktree / "colab_out" / "LOG.md"
    log.write_text(log.read_text() + f"| {done['finished_at']} | {job_id} | {done['status']} | "
                   f"{done['gpu_minutes']} | {main_number} | {next_job} |\n")
    run_git("add", "-A", cwd=worktree)
    run_git("commit", "-m", f"results({job_id}): {done['status']}", cwd=worktree)
    run_git("push", "origin", RESULTS_BRANCH, cwd=worktree)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--colab-out", default="colab_out")
    parser.add_argument("--worktree", default="/content/colab_results")
    parser.add_argument("--main-number", default="")
    parser.add_argument("--next-job", default="")
    args = parser.parse_args()
    worktree = ensure_worktree(Path(args.worktree))
    publish(args.job_id, Path(args.colab_out) / args.job_id, worktree, args.main_number, args.next_job)
    print(f"publicado: colab_out/{args.job_id} em {RESULTS_BRANCH}")


if __name__ == "__main__":
    main()
