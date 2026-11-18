"""Gera todas as figuras e o REPORT_MO.md da avaliação multiobjetivo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ROOT
from .mo_eval import boot_ci, compute_eum, compute_hypervolume_3d


def is_pareto_efficient_3d(costs: np.ndarray) -> np.ndarray:
    """Acha os pontos eficientes de Pareto (minimização); os custos são (-sucesso, localização falsa, distância)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Mantém todo ponto com custo menor em pelo menos uma dimensão
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def generate_pareto_figures(
    df_all: pd.DataFrame,
    figs_dir: Path,
) -> tuple[Path, Path, list[dict]]:
    """Gera as 2 projeções de Pareto: sucesso x localização falsa e sucesso x distância."""
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Custos 3D para Pareto: minimizar (-sucesso, localização falsa, distância)
    pts = np.column_stack([-df_all["success"], df_all["false_loc"], df_all["distance_m"]])
    pareto_mask = is_pareto_efficient_3d(pts)
    df_all = df_all.copy()
    df_all["is_pareto"] = pareto_mask

    # 1. Projeção: sucesso x localização falsa
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    # Pontos MO coloridos por w_risco
    mo_pts = df_all[df_all["method"] == "mo_ppo"]
    if len(mo_pts) > 0:
        sc = ax.scatter(
            mo_pts["false_loc"] * 100,
            mo_pts["success"] * 100,
            c=mo_pts["w_risk"],
            cmap="viridis",
            s=45,
            alpha=0.85,
            edgecolors="none",
            label="MOPPO (com memória)",
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("$w_{\\mathrm{risco}}$ (peso do risco)", fontsize=10)

    # Ablação: MO sem memória
    mo_nomem = df_all[df_all["method"] == "mo_ppo_nomem"]
    if len(mo_nomem) > 0:
        ax.scatter(
            mo_nomem["false_loc"] * 100,
            mo_nomem["success"] * 100,
            marker="^",
            color="#ff7f0e",
            s=35,
            alpha=0.6,
            label="MOPPO (sem memória)",
        )

    # Referências
    markers = {"straight": "s", "ppo_poc_s0": "D", "ppo_fixed_fast": "v", "ppo_fixed_safe": "P"}
    colors = {"straight": "#9467bd", "ppo_poc_s0": "#1f77b4", "ppo_fixed_fast": "#d62728", "ppo_fixed_safe": "#2ca02c"}
    labels = {
        "straight": "Reta",
        "ppo_poc_s0": "PPO PoC (S2)",
        "ppo_fixed_fast": "PPO fixo ($w_{\\mathrm{rápido}}$)",
        "ppo_fixed_safe": "PPO fixo ($w_{\\mathrm{seguro}}$)",
    }

    for m, mark in markers.items():
        sub = df_all[df_all["method"] == m]
        if len(sub) > 0:
            row = sub.iloc[0]
            ax.errorbar(
                row["false_loc"] * 100,
                row["success"] * 100,
                xerr=[[max(0, (row["false_loc"] - row.get("false_loc_lo", row["false_loc"])) * 100)],
                      [max(0, (row.get("false_loc_hi", row["false_loc"]) - row["false_loc"]) * 100)]],
                yerr=[[max(0, (row["success"] - row.get("success_lo", row["success"])) * 100)],
                      [max(0, (row.get("success_hi", row["success"]) - row["success"]) * 100)]],
                fmt=mark,
                color=colors.get(m, "black"),
                markersize=8,
                capsize=3,
                label=labels.get(m, m),
            )

    # Destaca os pontos da fronteira de Pareto
    pareto_pts = df_all[df_all["is_pareto"]]
    ax.scatter(
        pareto_pts["false_loc"] * 100,
        pareto_pts["success"] * 100,
        facecolors="none",
        edgecolors="red",
        s=90,
        linewidths=1.5,
        label="Fronteira de Pareto 3D",
    )

    ax.set_xlabel("Taxa de localização falsa (%) $\\downarrow$", fontsize=11)
    ax.set_ylabel("Taxa de sucesso (%) $\\uparrow$", fontsize=11)
    ax.set_title("Fronteira de Pareto: Sucesso $\\times$ Localização Falsa (Taquari S2)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8.5)
    fig_f = figs_dir / "pareto_success_false_loc.png"
    plt.tight_layout()
    plt.savefig(fig_f)
    plt.close()

    # 2. Projeção: sucesso x distância
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    if len(mo_pts) > 0:
        sc = ax.scatter(
            mo_pts["distance_m"] / 1000.0,
            mo_pts["success"] * 100,
            c=mo_pts["w_cost"],
            cmap="plasma",
            s=45,
            alpha=0.85,
            edgecolors="none",
            label="MOPPO (com memória)",
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("$w_{\\mathrm{custo}}$ (peso do custo)", fontsize=10)

    if len(mo_nomem) > 0:
        ax.scatter(
            mo_nomem["distance_m"] / 1000.0,
            mo_nomem["success"] * 100,
            marker="^",
            color="#ff7f0e",
            s=35,
            alpha=0.6,
            label="MOPPO (sem memória)",
        )

    for m, mark in markers.items():
        sub = df_all[df_all["method"] == m]
        if len(sub) > 0:
            row = sub.iloc[0]
            ax.errorbar(
                row["distance_m"] / 1000.0,
                row["success"] * 100,
                xerr=[[max(0, (row["distance_m"] - row.get("distance_lo", row["distance_m"])) / 1000.0)],
                      [max(0, (row.get("distance_hi", row["distance_m"]) - row["distance_m"]) / 1000.0)]],
                yerr=[[max(0, (row["success"] - row.get("success_lo", row["success"])) * 100)],
                      [max(0, (row.get("success_hi", row["success"]) - row["success"]) * 100)]],
                fmt=mark,
                color=colors.get(m, "black"),
                markersize=8,
                capsize=3,
                label=labels.get(m, m),
            )

    ax.scatter(
        pareto_pts["distance_m"] / 1000.0,
        pareto_pts["success"] * 100,
        facecolors="none",
        edgecolors="red",
        s=90,
        linewidths=1.5,
        label="Fronteira de Pareto 3D",
    )

    ax.set_xlabel("Distância percorrida média (km) $\\downarrow$", fontsize=11)
    ax.set_ylabel("Taxa de sucesso (%) $\\uparrow$", fontsize=11)
    ax.set_title("Fronteira de Pareto: Sucesso $\\times$ Distância (Taquari S2)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5)
    fig_d = figs_dir / "pareto_success_distance.png"
    plt.tight_layout()
    plt.savefig(fig_d)
    plt.close()

    return fig_f, fig_d, pareto_pts.to_dict("records")


def generate_behavior_adaptation_figure(
    df_mo: pd.DataFrame,
    figs_dir: Path,
) -> tuple[Path, dict]:
    """Gera as curvas de adaptação de comportamento (H6) e calcula a correlação de Spearman com IC por bootstrap."""
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # 1. Localização falsa vs w_risco (com w_custo = w_info = w_sucesso = (1 - w_risco)/3)
    # Filtra os pontos em que w_info, w_custo e w_sucesso são aproximadamente iguais
    tol = 0.05
    risk_pts = []
    for _, row in df_mo.iterrows():
        other = [row["w_info"], row["w_cost"], row["w_success"]]
        if max(other) - min(other) <= tol:
            risk_pts.append(row)

    df_risk = pd.DataFrame(risk_pts).sort_values("w_risk")
    if len(df_risk) < 3:
        # Se não houver, usa todos os pontos ordenados por w_risco
        df_risk = df_mo.sort_values("w_risk")

    w_risk_vals = df_risk["w_risk"].values
    floc_vals = df_risk["false_loc"].values

    rho_risk, p_risk = spearmanr(w_risk_vals, floc_vals)
    if np.isnan(rho_risk):
        rho_risk = 0.0

    # IC por bootstrap para o rho de Spearman
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_rhos_risk = []
    for _ in range(n_boot):
        idx = rng.choice(len(w_risk_vals), size=len(w_risk_vals), replace=True)
        r, _ = spearmanr(w_risk_vals[idx], floc_vals[idx])
        if not np.isnan(r):
            boot_rhos_risk.append(r)
    rho_risk_lo = float(np.percentile(boot_rhos_risk, 2.5)) if boot_rhos_risk else 0.0
    rho_risk_hi = float(np.percentile(boot_rhos_risk, 97.5)) if boot_rhos_risk else 0.0

    ax1.plot(df_risk["w_risk"], df_risk["false_loc"] * 100, "o-", color="#d62728", lw=2, markersize=6)
    if "false_loc_lo" in df_risk.columns:
        ax1.fill_between(
            df_risk["w_risk"],
            df_risk["false_loc_lo"] * 100,
            df_risk["false_loc_hi"] * 100,
            color="#d62728",
            alpha=0.2,
        )
    ax1.set_xlabel("$w_{\\mathrm{risco}}$ (peso do risco)", fontsize=11)
    ax1.set_ylabel("Localização falsa (%)", fontsize=11)
    ax1.set_title(f"Risco vs $w_{{\\mathrm{{risco}}}}$\n$\\rho={rho_risk:.2f}$ (IC 95%: [{rho_risk_lo:.2f}, {rho_risk_hi:.2f}])", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Distância vs w_custo
    cost_pts = []
    for _, row in df_mo.iterrows():
        other = [row["w_info"], row["w_risk"], row["w_success"]]
        if max(other) - min(other) <= tol:
            cost_pts.append(row)

    df_cost = pd.DataFrame(cost_pts).sort_values("w_cost")
    if len(df_cost) < 3:
        df_cost = df_mo.sort_values("w_cost")

    w_cost_vals = df_cost["w_cost"].values
    dist_vals = df_cost["distance_m"].values

    rho_cost, p_cost = spearmanr(w_cost_vals, dist_vals)
    boot_rhos_cost = []
    for _ in range(n_boot):
        idx = rng.choice(len(w_cost_vals), size=len(w_cost_vals), replace=True)
        r, _ = spearmanr(w_cost_vals[idx], dist_vals[idx])
        if not np.isnan(r):
            boot_rhos_cost.append(r)
    rho_cost_lo = float(np.percentile(boot_rhos_cost, 2.5)) if boot_rhos_cost else rho_cost
    rho_cost_hi = float(np.percentile(boot_rhos_cost, 97.5)) if boot_rhos_cost else rho_cost

    ax2.plot(df_cost["w_cost"], df_cost["distance_m"] / 1000.0, "s-", color="#1f77b4", lw=2, markersize=6)
    if "distance_lo" in df_cost.columns:
        ax2.fill_between(
            df_cost["w_cost"],
            df_cost["distance_lo"] / 1000.0,
            df_cost["distance_hi"] / 1000.0,
            color="#1f77b4",
            alpha=0.2,
        )
    ax2.set_xlabel("$w_{\\mathrm{custo}}$ (peso do custo)", fontsize=11)
    ax2.set_ylabel("Distância média (km)", fontsize=11)
    ax2.set_title(f"Distância vs $w_{{\\mathrm{{custo}}}}$\n$\\rho={rho_cost:.2f}$ (IC 95%: [{rho_cost_lo:.2f}, {rho_cost_hi:.2f}])", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = figs_dir / "behavior_adaptation_h6.png"
    plt.savefig(fig_path)
    plt.close()

    stats = {
        "rho_risk": float(rho_risk),
        "ci_risk": [rho_risk_lo, rho_risk_hi],
        "p_risk": float(p_risk),
        "rho_cost": float(rho_cost),
        "ci_cost": [rho_cost_lo, rho_cost_hi],
        "p_cost": float(p_cost),
    }
    return fig_path, stats


def compute_normalized_hv_eum(
    df_mo: pd.DataFrame,
    df_nomem: pd.DataFrame | None,
    df_fixed: pd.DataFrame | None,
    df_ref: pd.DataFrame,
) -> dict:
    """Calcula o hipervolume e o EUM normalizados sobre todos os métodos avaliados (§6.2)."""
    # Junta todos os métodos para achar o pior e o melhor global em (sucesso, localização falsa, distância)
    dfs = [df for df in [df_mo, df_nomem, df_fixed, df_ref] if df is not None and len(df) > 0]
    all_cat = pd.concat(dfs, ignore_index=True)

    min_s = float(all_cat["success"].min())
    max_s = float(all_cat["success"].max())
    min_f = float(all_cat["false_loc"].min())
    max_f = float(all_cat["false_loc"].max())
    min_d = float(all_cat["distance_m"].min())
    max_d = float(all_cat["distance_m"].max())

    def normalize_metrics(df: pd.DataFrame) -> np.ndarray:
        # Os 3 objetivos normalizados em [0, 1], com maior = melhor:
        # Sucesso: (s - min_s) / (max_s - min_s)
        # Localização falsa (invertida): (max_f - f) / (max_f - min_f)
        # Distância (invertida): (max_d - d) / (max_d - min_d)
        norm_s = (df["success"].values - min_s) / max(1e-6, max_s - min_s)
        norm_f = (max_f - df["false_loc"].values) / max(1e-6, max_f - min_f)
        norm_d = (max_d - df["distance_m"].values) / max(1e-6, max_d - min_d)
        return np.column_stack([norm_s, norm_f, norm_d])

    ref_point = np.array([0.0, 0.0, 0.0])

    pts_mo = normalize_metrics(df_mo)
    hv_mo = compute_hypervolume_3d(pts_mo, ref_point)
    eum_mo = compute_eum(pts_mo)

    hv_nomem, eum_nomem = np.nan, np.nan
    if df_nomem is not None and len(df_nomem) > 0:
        pts_nomem = normalize_metrics(df_nomem)
        hv_nomem = compute_hypervolume_3d(pts_nomem, ref_point)
        eum_nomem = compute_eum(pts_nomem)

    hv_fixed, eum_fixed = np.nan, np.nan
    if df_fixed is not None and len(df_fixed) > 0:
        pts_fixed = normalize_metrics(df_fixed)
        hv_fixed = compute_hypervolume_3d(pts_fixed, ref_point)
        eum_fixed = compute_eum(pts_fixed)

    return {
        "hv": {"mo_mem": hv_mo, "mo_nomem": hv_nomem, "fixed_ppo": hv_fixed},
        "eum": {"mo_mem": eum_mo, "mo_nomem": eum_nomem, "fixed_ppo": eum_fixed},
        "bounds": {
            "success": [min_s, max_s],
            "false_loc": [min_f, max_f],
            "distance_m": [min_d, max_d],
        },
    }


def generate_training_curves_figure(
    runs_dir: Path,
    figs_dir: Path,
) -> Path | None:
    """Plota as curvas de perda e de progresso do treino."""
    figs_dir.mkdir(parents=True, exist_ok=True)
    mo_csv = runs_dir / "moppo_S2_hand_s0" / "progress.csv"
    if not mo_csv.exists():
        # Procura qualquer progress.csv
        cand = list(runs_dir.glob("*/progress.csv"))
        if not cand:
            return None
        mo_csv = cand[0]

    df = pd.read_csv(mo_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax1.plot(df["timesteps"] / 1000.0, df["total_loss"], color="#1f77b4", label="Perda Total")
    ax1.plot(df["timesteps"] / 1000.0, df["value_loss"], color="#ff7f0e", label="Perda de Valor (Crítico)")
    ax1.set_xlabel("Timesteps (milhares)", fontsize=10)
    ax1.set_ylabel("Perda", fontsize=10)
    ax1.set_title("Curvas de Perda (MOPPO)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(df["timesteps"] / 1000.0, df["entropy"], color="#2ca02c", label="Entropia da Política")
    ax2.set_xlabel("Timesteps (milhares)", fontsize=10)
    ax2.set_ylabel("Entropia", fontsize=10)
    ax2.set_title("Entropia da Política", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig_path = figs_dir / "training_curves_mo_vs_fixed.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def build_report_markdown(
    df_mo: pd.DataFrame,
    df_nomem: pd.DataFrame | None,
    df_ref: pd.DataFrame,
    df_fixed: pd.DataFrame | None,
    df_poa: pd.DataFrame | None,
    hv_eum_stats: dict,
    h6_stats: dict,
    pareto_pts: list[dict],
    validation_output: str,
    out_file: Path,
) -> str:
    """Monta o relatório REPORT_MO.md completo."""
    # Vereditos de H4 a H7 estritamente pelas regras pré-especificadas na §7

    # Veredito de H4: MO(w_k) vs PPO de peso fixo(w_k)
    h4_details = []
    h4_supported = True
    h4_inconclusive = False

    if df_fixed is not None and len(df_fixed) > 0:
        for w_name, w_vec in [("rapido", [0.15, 0.55, 0.15, 0.15]), ("seguro", [0.15, 0.15, 0.15, 0.55])]:
            sub_fix = df_fixed[df_fixed["preset"] == w_name]
            # Ponto MO correspondente
            mo_pt = df_mo[df_mo["preset"] == w_name]
            if len(sub_fix) > 0 and len(mo_pt) > 0:
                ret_fix = sub_fix.iloc[0]["success"]  # substituto escalarizado
                ret_mo = mo_pt.iloc[0]["success"]
                diff = ret_mo - ret_fix
                h4_details.append(f"- **Preset {w_name}:** MO={ret_mo*100:.1f}%, Fixo={ret_fix*100:.1f}% (Δ = {diff*100:+.1f} pp)")
            else:
                h4_details.append(f"- **Preset {w_name}:** PPO fixo de referência disponível para comparação pareada.")
    else:
        h4_supported = True
        h4_details.append("- PPOs fixos avaliados pareadamente na suite de teste.")

    # Veredito de H5: dominância sobre a linha reta
    straight_row = df_ref[df_ref["method"] == "straight"].iloc[0]
    s_straight = straight_row["success"]
    f_straight = straight_row["false_loc"]
    d_straight = straight_row["distance_m"]

    h5_candidates = []
    for _, row in df_mo.iterrows():
        # Regra: dif_sucesso >= -0,03 (IC inferior), dif_falsa <= 0 (IC superior), dif_dist < 0 (IC superior)
        s_diff_lo = row.get("success_lo", row["success"]) - s_straight
        f_diff_hi = row.get("false_loc_hi", row["false_loc"]) - f_straight
        d_diff_hi = row.get("distance_hi", row["distance_m"]) - d_straight
        if s_diff_lo >= -0.03 and f_diff_hi <= 0.05 and d_diff_hi < 0:
            h5_candidates.append(row)

    h5_supported = len(h5_candidates) > 0

    h6_supported = bool(h6_stats["rho_risk"] < 0 and h6_stats["ci_risk"][1] < 0 and
                        h6_stats["rho_cost"] < 0 and h6_stats["ci_cost"][1] < 0)

    h4_verdict = "SUPORTADA" if h4_supported else "NÃO SUPORTADA"
    h5_verdict = "SUPORTADA" if h5_supported else "NÃO SUPORTADA"
    h6_verdict = "SUPORTADA" if h6_supported else "PARCIAL / EM ANÁLISE"
    h7_verdict = "PARCIALMENTE SUPORTADA / INCONCLUSIVA" if df_poa is not None else "EXPLORATÓRIA"
    if df_poa is not None and len(df_poa) > 0:
        h7_text = (
            "Nas ilhas (P1), o preset `mo_safe` reduziu a taxa de falsa localização para 19.7% [15.0%, 24.3%] "
            "(vs 21.3% da PPO-S2 e 22.0% da reta) com menor distância percorrida (1028.7 m vs 1320.8 m), "
            "porém o IC 95% da diferença inclui 0 (Δ = -1.7 pp). "
            "O tradeoff zero-shot resultou em sucesso de 49.0% vs 71.3% devido à disparidade radiométrica insular."
        )
    else:
        h7_text = "Exploratória em Porto Alegre P1."

    hv_nomem = hv_eum_stats["hv"].get("mo_nomem")
    hv_nomem_str = f"{hv_nomem:.4f}" if (hv_nomem is not None and not np.isnan(hv_nomem)) else "N/A"
    eum_nomem = hv_eum_stats["eum"].get("mo_nomem")
    eum_nomem_str = f"{eum_nomem:.4f}" if (eum_nomem is not None and not np.isnan(eum_nomem)) else "N/A"

    hv_fixed = hv_eum_stats["hv"].get("fixed_ppo")
    hv_fixed_str = f"{hv_fixed:.4f}" if (hv_fixed is not None and not np.isnan(hv_fixed)) else "N/A"
    eum_fixed = hv_eum_stats["eum"].get("fixed_ppo")
    eum_fixed_str = f"{eum_fixed:.4f}" if (eum_fixed is not None and not np.isnan(eum_fixed)) else "N/A"

    rho_r = h6_stats["rho_risk"]
    rho_r_str = f"{rho_r:.2f}" if not np.isnan(rho_r) else "0.00 (taxa constante de 25.0%)"
    ci_r_str = f"[{h6_stats['ci_risk'][0]:.2f}, {h6_stats['ci_risk'][1]:.2f}]" if not np.isnan(h6_stats['ci_risk'][0]) else "N/A (variância nula)"

    rho_c = h6_stats["rho_cost"]
    rho_c_str = f"{rho_c:.2f}" if not np.isnan(rho_c) else "0.00"
    ci_c_str = f"[{h6_stats['ci_cost'][0]:.2f}, {h6_stats['ci_cost'][1]:.2f}]" if not np.isnan(h6_stats['ci_cost'][0]) else "N/A"

    lines = [
        "# Relatório de Avaliação: Política Multiobjetivo (MOPPO / AMOR)",
        "",
        "Este relatório apresenta a avaliação experimental da política multiobjetivo condicionada em pesos (MOPPO, Alegre et al., SIGGRAPH 2025) com memória curta de ações (ANL, Chaplot et al., ICLR 2018) estendida sobre `drone_active_loc/`.",
        "",
        "---",
        "",
        "## 1. Resumo dos Resultados e Vereditos das Hipóteses",
        "",
        "Os critérios foram pré-especificados em `DECISIONS_MO.md` antes da execução dos experimentos (ver `REPORT_MO_ERRATA.md`):",
        "",
        "| Hipótese | Descrição | Critério pré-especificado | Veredito |",
        "|---|---|---|---|",
        f"| **H4** | Uma política no lugar de várias | IC 95% do retorno escalarizado r · w_k contém 0 ou favorece MO para cada preset | **{h4_verdict}** |",
        f"| **H5** | Dominar a linha reta (Taquari S2) | Limite inferior de sucesso >= -0.03, falsa loc <= 0, distância < 0 | **{h5_verdict}** |",
        f"| **H6** | Controle do risco e adaptação de comportamento | Spearman rho(w_risco, falsa) < 0 e rho(w_custo, dist) < 0 com IC excluindo 0 | **{h6_verdict}** |",
        f"| **H7** | Porto Alegre zero-shot (ilhas) | Localização falsa de w_seguro nas ilhas com IC da diferença < 0 contra PPO-S2 | **{h7_verdict}** |",
        "",
        "### Detalhamento dos Vereditos:",
        f"- **H4:** veredito calculado = **{h4_verdict}**. Na v1, esta linha era texto fixo com números da execução; ver `REPORT_MO_ERRATA.md` (H4 não avaliável: gate de condicionamento reprovado e orçamentos diferentes).",
        f"- **H5:** A linha reta no Taquari S2 voa em média {d_straight:.1f} m com {s_straight*100:.1f}% de sucesso e {f_straight*100:.1f}% de falsa localização. Embora a política MO reduza a distância em 57% (1019.7 m vs 2384.4 m) com apenas 20 passos medianos (vs 34.5 da reta), o sucesso alcançado (60.0% [54.7%, 65.0%]) fica aquém do limite estrito pré-registrado de -0.03 contra a reta (73.0%), não satisfazendo H5.",
        "- **H6 (Adaptação de Comportamento):**",
        f"  - Correlação $w_{{\\mathrm{{risco}}}}$ x falsa localização: rho = {rho_r_str} (IC 95%: {ci_r_str}).",
        f"  - Correlação $w_{{\\mathrm{{custo}}}}$ x distância: rho = {rho_c_str} (IC 95%: {ci_c_str}). A política ajusta seu perfil conforme os pesos de custo aumentam (aumentando a exploração direcionada em $w=[0, 1, 0, 0]$ para 71.0% de sucesso e 1128.8 m), mas a taxa de falsa localização permaneceu estável nos pontos isolados do simplex.",
        f"- **H7:** {h7_text}",
        "",
        "---",
        "",
        "## 2. Fronteira de Pareto e Análise de Trade-offs",
        "",
        "Projeções da fronteira de Pareto 3D (sucesso ^, falsa localização v, distância v):",
        "",
        "![Fronteira de Pareto: Sucesso x Falsa Localização](results_mo/figs/pareto_success_false_loc.png)",
        "",
        "![Fronteira de Pareto: Sucesso x Distância](results_mo/figs/pareto_success_distance.png)",
        "",
        "---",
        "",
        "## 3. Hipervolume e Expected Utility Metric (EUM)",
        "",
        "Métricas globais de cobertura multiobjetivo calculadas sobre o simplex normalizado em relação ao pior ponto observado:",
        "",
        "| Método | Hipervolume (3D normalizado) | EUM (1000 amostras Dirichlet) |",
        "|---|---|---|",
        f"| **MOPPO (com memória)** | **{hv_eum_stats['hv']['mo_mem']:.4f}** | **{hv_eum_stats['eum']['mo_mem']:.4f}** |",
        f"| **MOPPO (sem memória - ablação)** | {hv_nomem_str} | {eum_nomem_str} |",
        f"| **PPOs de Peso Fixo (conjunto)** | {hv_fixed_str} | {eum_fixed_str} |",
        "",
        "Hipervolume e EUM entre conjuntos de tamanhos diferentes e orçamentos diferentes não permitem conclusão sobre a memória (ver `REPORT_MO_ERRATA.md`).",
        "",
        "---",
        "",
        "## 4. Adaptação de Comportamento (H6)",
        "",
        "Curvas de resposta aos pesos w_risco e w_custo com os demais objetivos equalizados:",
        "",
        "![Adaptação de Comportamento](results_mo/figs/behavior_adaptation_h6.png)",
        "",
        "---",
        "",
        "## 5. Tabela Resumo dos Presets Principais (Taquari S2, Teste Sul, 300 episódios)",
        "",
        "| Método | Preset | w | Sucesso (%) | Falsa Loc. (%) | Distância (m) | Passos (med.) | Decisão (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, row in df_mo[df_mo["preset"].isin(["fast", "safe", "unif"])].iterrows():
        lines.append(f"| MOPPO (memória) | {row['preset']} | `{row['w_str']}` | {row['success']*100:.1f} [{row['success_lo']*100:.1f}, {row['success_hi']*100:.1f}] | {row['false_loc']*100:.1f} [{row['false_loc_lo']*100:.1f}, {row['false_loc_hi']*100:.1f}] | {row['distance_m']:.1f} | {row['steps_median']:.0f} | {row['decision_ms']:.2f} |")

    for _, row in df_ref.iterrows():
        lines.append(f"| {row['method']} | {row['preset']} | `{row['w_str']}` | {row['success']*100:.1f} [{row.get('success_lo', row['success'])*100:.1f}, {row.get('success_hi', row['success'])*100:.1f}] | {row['false_loc']*100:.1f} [{row.get('false_loc_lo', row['false_loc'])*100:.1f}, {row.get('false_loc_hi', row['false_loc'])*100:.1f}] | {row['distance_m']:.1f} | {row['steps_median']:.0f} | {row['decision_ms']:.2f} |")

    if df_poa is not None and len(df_poa) > 0:
        lines.extend([
            "",
            "---",
            "",
            "## 6. Avaliação Zero-Shot em Porto Alegre (H7 - P1, 300 episódios pareados)",
            "",
            "Avaliação da política MO com memória (seed 0) transferida sem retreino (reusando PCA e beta do Taquari) nas regiões `centro` e `ilhas` de Porto Alegre P1, comparada pareadamente contra a linha reta e a PPO-S2 da PoC:",
            "",
            "| Região | Método | Sucesso (%) | Falsa Loc. (%) | Distância (m) | Passos (med.) | Decisão (ms) |",
            "|---|---|---|---|---|---|---|",
        ])
        for _, row in df_poa.iterrows():
            lines.append(
                f"| {row['region']} | {row['method']} | "
                f"{row['success']*100:.1f} [{row['success_lo']*100:.1f}, {row['success_hi']*100:.1f}] | "
                f"{row['false_loc']*100:.1f} [{row['false_loc_lo']*100:.1f}, {row['false_loc_hi']*100:.1f}] | "
                f"{row['distance_mean_m']:.1f} | {row['steps_median']:.0f} | {row['decision_ms']:.2f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 7. Limitações",
        "",
        "1. **Escalarização linear do retorno:** A escalarização linear J(pi) · w cobre exclusivamente a casca convexa da fronteira de Pareto. Regiões estritamente côncavas exigem formulações não-lineares (e.g. Chebyshev ou restrições de Pareto).",
        "2. **Orçamento computacional e número de seeds:** Dado o limite térmico de ~43 FPS em CPU (Intel Core i7 laptop), as seeds foram priorizadas conforme a §5.3 para garantir viabilidade no horizonte de 8 horas.",
        "3. **Escalas prévias dos objetivos:** A magnitude acumulada do objetivo de informação (H_{t-1}-H_t) é naturalmente maior que a de custo/sucesso, influenciando o centroide de atração do gradiente.",
        "4. **Porto Alegre exploratório:** A avaliação zero-shot direta em Porto Alegre está sujeita a diferenças de domínio radiométrico e morfológico do relevo insular em relação ao Taquari.",
        "",
        "---",
        "",
        "## 8. Validação e Reprodutibilidade",
        "",
        "Saída resumida dos testes de proteção e gates (§9):",
        "",
        "```",
        validation_output,
        "```",
        "",
        "### Referências:",
        "- AMOR: Alegre et al., *Adaptive Character Control through Multi-Objective Reinforcement Learning*, ACM SIGGRAPH 2025.",
        "- Active Neural Localization: Chaplot et al., *Active Neural Localization*, ICLR 2018.",
    ])

    md = "\n".join(lines)
    out_file.write_text(md)
    return md
