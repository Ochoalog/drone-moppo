"""Vereditos H2–H5 da v3 (§2.2 do DECISIONS_V3.md), gerados por código a partir dos `episodes.csv` da avaliação.

Para cada modo de avaliação das políticas (argmax, amostrado) e cada critério do gate de aprendizado (emendado
em V38 = todas as execuções válidas; original = só as que passaram na vantagem de sucesso ≥ 5 p.p.).
Entradas: `results_v3/eval/<job>/episodes.csv` (eval_A obrigatório; eval_B e eval_C opcionais).
Saídas: `results_v3/verdicts/verdicts_v3.json` e `results_v3/verdicts/h3_places.csv`.
Uso: .venv/bin/python scripts/verdicts_v3.py
"""
from __future__ import annotations

import json

import pandas as pd

from dal.config import ROOT
from dal.v3.evaluation import verdicts
from dal.v3.reporting.provenance import git_commit, with_provenance

EVAL = ROOT / "results_v3" / "eval"
OUT = ROOT / "results_v3" / "verdicts"
PLACES = {"taquari": ("taquari", "S2"), "poa_ilhas": ("poa_ilhas", "P1"), "poa_centro": ("poa_centro", "P1"),
          "canoas": ("canoas", "C1")}
H2_PLACES = ("taquari", "poa_ilhas")


def select(episodes: pd.DataFrame, place: str, split: str = "test") -> pd.DataFrame:
    region, scenario = PLACES[place]
    return episodes[(episodes.region == region) & (episodes.scenario == scenario) & (episodes.split == split)]


def view(episodes: pd.DataFrame, mode: str, gate: str) -> pd.DataFrame:
    """Baselines (modo "—") + políticas no modo pedido; no critério original, só execuções que passaram nele."""
    baselines = episodes[episodes["mode"] == "—"]
    policies = episodes[episodes["mode"] == mode]
    if gate == "original":
        policies = policies[policies.gate_original_passed.astype(bool)]
    return pd.concat([baselines, policies], ignore_index=True)


def run(episodes: pd.DataFrame, mode: str, gate: str) -> tuple[dict, pd.DataFrame]:
    data = view(episodes, mode, gate)
    heuristic = verdicts.best_ca_heuristic(select(data, "taquari", "train"))
    result = {"mode": mode, "gate_criterion": gate, "best_ca_heuristic": heuristic,
              "runs": data[data.model_seed >= 0].groupby("method").model_seed.nunique().to_dict()}
    present = set(data.method)
    if {"straight_ca", "straight", "ppo_ca", "ppo_ca_stdfilter"} <= present:
        h2 = verdicts.verdict_h2({p: select(data, p) for p in H2_PLACES})
        result["H2"] = {"verdict": h2["verdict"], "table": h2["table"].to_dict("records")}
    if "ppo_ca" in present:
        h3 = verdicts.verdict_h3({p: verdicts.verdict_h3_for_place(select(data, p), "ppo_ca", heuristic)
                                  for p in PLACES if not select(data, p).empty})
        result["H3"] = {"by_place": h3, "taquari_supported": h3.get("taquari", {}).get("supported_holm")}
        result["H5"] = verdicts.verdict_h5(h3)
    if {"ppo_ca", "ppo_ca_alt"} <= present:
        result["H4"] = verdicts.verdict_h4(select(data, "taquari"))
    table = pd.DataFrame([{"place": p, "mode": mode, "gate_criterion": gate, **r}
                          for p, r in result.get("H3", {}).get("by_place", {}).items()])
    return result, table


def main() -> None:
    frames = [pd.read_csv(path) for path in sorted(EVAL.glob("eval_A*/episodes.csv"))]
    if not frames:
        raise SystemExit(f"sem episódios em {EVAL}/eval_A*/")
    episodes = pd.concat(frames, ignore_index=True)
    results, tables = [], []
    for mode in ("argmax", "sampled"):
        for gate in ("emended", "original"):
            result, table = run(episodes, mode, gate)
            results.append(result)
            tables.append(table)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "source_file": [str(p) for p in sorted(EVAL.glob("eval_A*/episodes.csv"))],
               "protocol": "§2 do DECISIONS_V3.md; 1000 episódios pareados (seeds 2024+i); bootstrap B = 2000",
               "git_commit": git_commit()}
    (OUT / "verdicts_v3.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
    h3 = with_provenance(pd.concat(tables, ignore_index=True), source_file="results_v3/eval/eval_A/episodes.csv",
                         protocol="H3 por local, Holm entre locais", config={"modes": ["argmax", "sampled"]},
                         n_episodes=1000, seeds="0–9", fold="A", region="todas")
    h3.to_csv(OUT / "h3_places.csv", index=False)
    for result in results:
        print(result["mode"], result["gate_criterion"], {k: result[k]["verdict"] for k in ("H2", "H4", "H5")
                                                          if k in result})


if __name__ == "__main__":
    main()
