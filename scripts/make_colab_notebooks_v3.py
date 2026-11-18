# ruff: noqa: E501  (código das células como texto)
"""Gera os notebooks de colab/ (fonte única: este script). Uso: .venv/bin/python scripts/make_colab_notebooks_v3.py"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from dal.config import ROOT

REPO = "https://github.com/Ochoalog/drone-moppo.git"
CODE_BRANCH = "feat/v3-change-aware"

SETUP_CELLS = [
    new_markdown_cell("# 00 · Preparação da sessão (IA-2)\n\nMonta o Drive, clona o repositório **no commit de "
                      "`colab/CURRENT_COMMIT`**, instala as dependências e testa a escrita na branch `colab/results`. "
                      "Rode no início de toda sessão. O token vem dos *secrets* do Colab (`GITHUB_TOKEN`)."),
    new_code_cell("from google.colab import drive, userdata\n"
                  "drive.mount('/content/drive')\n"
                  "import os, pathlib\n"
                  "DRIVE = pathlib.Path('/content/drive/MyDrive/drone_v3'); DRIVE.mkdir(parents=True, exist_ok=True)\n"
                  "TOKEN = userdata.get('GITHUB_TOKEN')  # se falhar: pedir ao Vitor para criar o secret\n"
                  "os.environ['GITHUB_TOKEN'] = TOKEN"),
    new_code_cell("%%bash\nset -e\ncd /content\n"
                  f"if [ ! -d repo ]; then git clone https://x-access-token:$GITHUB_TOKEN@{REPO[8:]} repo; fi\n"
                  "cd repo\ngit config user.name 'IA-2 (Colab)'\ngit config user.email 'ia2-colab@users.noreply.github.com'\n"
                  f"git fetch origin {CODE_BRANCH} colab/results\n"
                  f"COMMIT=$(git show origin/{CODE_BRANCH}:colab/CURRENT_COMMIT | tr -d '[:space:]')\n"
                  "git checkout -q $COMMIT\necho \"código no commit $COMMIT\"\ngit log --oneline -1"),
    new_code_cell("%%bash\nset -e\ncd /content/repo\n"
                  "pip -q install -e . timm rliable pystac-client planetary-computer geopandas pyarrow nbformat\n"
                  "nvidia-smi --query-gpu=name,memory.total --format=csv"),
    new_code_cell("%%bash\nset -e\ncd /content/repo\n"
                  "python scripts/colab_publish_v3.py --help > /dev/null\n"
                  "mkdir -p colab_out/_setup_check && date -u > colab_out/_setup_check/ping.txt\n"
                  "python - <<'EOF'\nfrom pathlib import Path\nfrom dal.v3.reporting.colab_io import write_done\n"
                  "write_done(Path('colab_out/_setup_check'), job_id='_setup_check', config={}, started_at='-', gpu_minutes=0,"
                  " metrics={'ping': True})\nEOF\n"
                  "python scripts/colab_publish_v3.py --job-id _setup_check --main-number 'escrita ok'\n"),
]


G1_EVAL_CELL = new_code_cell(
    "# G1 preliminar: recuperação com a mesma função do REPORT.md §2 (window_stats), no próprio Colab.\n"
    "import json, yaml\n"
    "cfg = yaml.safe_load(open(f'/content/repo/{JOB_CONFIG}'))\n"
    "base = f\"/content/drive/MyDrive/drone_v3/{cfg['output_dir']}\"\n"
    "for name in ('gem', 'cls', 'hand'):\n"
    "    subprocess.run([sys.executable, 'scripts/eval_retrieval_v3.py', '--spec', cfg['spec'], '--descriptors',\n"
    "                    f'{base}/{name}.npy', '--name', f'{JOB_ID}_{name}', '--out', f'colab_out/{JOB_ID}'],\n"
    "                   cwd='/content/repo', check=True)")


def job_notebook(title: str, description: str, module_command: str, default_job: str,
                 extra_cells: list | None = None) -> nbformat.NotebookNode:
    return new_notebook(cells=[
        new_markdown_cell(f"# {title}\n\n{description}\n\nAntes: rode `00_setup.ipynb`. O job é **retomável**: se a "
                          "sessão cair, rode o setup e este notebook de novo com o mesmo `JOB_ID`."),
        new_code_cell(f"JOB_ID = '{default_job}'\n"
                      "import pandas as pd\njobs = pd.read_csv('/content/repo/colab/jobs_v3.csv')\n"
                      "job = jobs.set_index('job_id').loc[JOB_ID]\nJOB_CONFIG = job.config_path\nprint(job)"),
        new_code_cell("import subprocess, sys\n"
                      f"cmd = [sys.executable, '-m', {module_command!r}, '--job-config', JOB_CONFIG, '--job-id', JOB_ID,\n"
                      "       '--out-root', '/content/drive/MyDrive/drone_v3', '--colab-out', 'colab_out']\n"
                      "print(subprocess.run(cmd, cwd='/content/repo', check=True).returncode)"),
        *(extra_cells or []),
        new_code_cell("import json\ndone = json.load(open(f'/content/repo/colab_out/{JOB_ID}/DONE.json'))\nprint(json.dumps(done, indent=1))\n"
                      "main_number = str(done['metrics'].get('crops_per_second', ''))[:12]\n"
                      "subprocess.run([sys.executable, 'scripts/colab_publish_v3.py', '--job-id', JOB_ID,\n"
                      "                '--main-number', f'crops/s {main_number}'], cwd='/content/repo', check=True)"),
    ])


def main() -> None:
    notebooks = {
        "00_setup.ipynb": new_notebook(cells=SETUP_CELLS),
        "01_data_and_crops.ipynb": job_notebook(
            "01 · Specs v3 e descritores multialtitude",
            "Gera as specs v3 do local (grade de 25 m, K = 4 variantes) com `dal.v3.sensor.specs` e **confere os hashes "
            "contra `colab/specs/v3_manifest.json`** (se não baterem, o job para). Depois extrai DINOv2 (GeM e CLS) e a "
            "qualidade de cada recorte (fração válida e de nuvem) em `MyDrive/drone_v3/features/v3/<modelo>/<spec>/`.",
            "dal.v3.descriptors.extract_site", "v3_extract_taquari_100m"),
        "03_contrastive_finetune.ipynb": job_notebook(
            "03 · Ajuste fino contrastivo (InfoNCE)",
            "Ajusta os últimos blocos do DINOv2 com pares (mapa pré-evento, variante de observação pós-evento) da mesma "
            "célula, **só na faixa de treino da dobra** do Taquari. Checkpoint no Drive a cada ≤ 15 min; se o tempo do "
            "job acabar, o `DONE.json` sai com `FAILED` e é só rodar de novo (retoma). No fim grava "
            "`MyDrive/drone_v3/finetune/<job_id>/backbone.pt`.",
            "dal.v3.descriptors.finetune", "finetune_A"),
        "02_dinov2_features.ipynb": job_notebook(
            "02 · Descritores DINOv2 a partir de uma spec",
            "Renderiza os recortes da spec na GPU (`dal.v3.sensor.render`), extrai DINOv2 (GeM e CLS) e, se a config "
            "pedir, o hand da v1 nos mesmos pixels. Grava `float16 [N, D]` em `MyDrive/drone_v3/<output_dir>/` com "
            "`MANIFEST.sha256`, e o resumo em `colab_out/<job_id>/`.",
            "dal.v3.descriptors.extract", "g1_legacy_dinov2", extra_cells=[G1_EVAL_CELL]),
    }
    for name, notebook in notebooks.items():
        notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
        notebook.metadata["accelerator"] = "GPU"
        nbformat.write(notebook, ROOT / "colab" / name)
        print("escrito", name)


if __name__ == "__main__":
    main()
