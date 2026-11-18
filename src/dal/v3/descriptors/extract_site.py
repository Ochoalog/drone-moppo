"""Job do notebook 01: gera as specs v3 de um local (conferindo o manifesto) e extrai os descritores.

Para cada altitude pedida: gera as specs (`dal.v3.sensor.specs`, com `--verify` contra o manifesto
commitado) e roda `dal.v3.descriptors.extract` em cada uma, gravando em
`<out_root>/features/v3/<modelo>/<spec>/` (gem, cls, valid_frac, cloud_frac). Retomável: as specs já
extraídas são puladas pelos blocos do próprio extrator.

Uso (Colab): python -m dal.v3.descriptors.extract_site --job-config configs_v3/jobs/v3_extract_taquari_100m.yaml \
    --job-id v3_extract_taquari_100m --out-root /content/drive/MyDrive/drone_v3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dal.config import ROOT
from dal.v3.descriptors import extract
from dal.v3.reporting.colab_io import load_job_config, now_iso, write_done
from dal.v3.sensor import specs


def run(config: dict, job_id: str, out_root: Path, colab_out: Path, device: str, spec_dir: Path) -> dict:
    started_at, started = now_iso(), time.time()
    site = specs.load_site(config["site"])
    manifest = json.loads((ROOT / "colab" / "specs" / specs.MANIFEST).read_text())
    spec_dir.mkdir(parents=True, exist_ok=True)
    # `model` nomeia a pasta de saída; `hub_model` é o modelo base do torch.hub (padrão: o próprio `model`).
    hub_model = config.get("hub_model", config["model"])
    # Saídas configuráveis: o padrão das specs v3 é só o GeM (o CLS dobraria o espaço no Drive sem uso).
    extractor = extract.DinoV2Extractor(hub_model, int(config.get("input_px", 224)), device,
                                        weights=config.get("weights"), grids=tuple(config.get("grids", ())),
                                        outputs=tuple(config.get("outputs", ("gem",))))
    done = {}
    for name, frame in specs.iter_specs(site):
        if int(name.rsplit("_", 1)[1].rstrip("m")) not in config["altitudes"]:
            continue
        digest = specs.content_hash(frame)
        if manifest.get(name) != digest:
            raise SystemExit(f"{name}: hash {digest} ≠ manifesto {manifest.get(name)} (código diferente do commitado?)")
        path = spec_dir / f"{name}.parquet"
        if not path.exists():
            frame.to_parquet(path, index=False)
        job = {"spec": str(path), "output_dir": f"features/v3/{config['model']}/{name}",
               "chunk_rows": int(config.get("chunk_rows", 50000)), "batch_size": int(config.get("batch_size", 256)),
               "margin_m": 200.0, "model": config["model"]}
        metrics = extract.run(job, f"{job_id}__{name}", out_root, colab_out / "_parts", device, extractor=extractor)
        done[name] = {"rows": metrics["n_rows"], "crops_per_second": metrics["crops_per_second"]}
        print(f"[extract_site] {name}: {done[name]}", flush=True)
    minutes = (time.time() - started) / 60
    metrics = {"specs": done, "minutes": minutes}
    write_done(colab_out / job_id, job_id=job_id, config=config, started_at=started_at,
               gpu_minutes=minutes if device.startswith("cuda") else 0.0, metrics=metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--colab-out", default="colab_out")
    parser.add_argument("--spec-dir", default="/content/specs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_job_config(ROOT / args.job_config)
    print(run(config, args.job_id, Path(args.out_root), Path(args.colab_out), args.device, Path(args.spec_dir)))


if __name__ == "__main__":
    main()
