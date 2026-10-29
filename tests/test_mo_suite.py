"""Test suite for Multi-Objective extensions (MOPPO, MOEnv, Hypervolume, Protection)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from dal.config import ROOT
from dal.env import DroneLocEnv
from dal.mo_env import DroneLocMOEnv, make_mo_env
from dal.mo_eval import compute_hypervolume_2d, compute_hypervolume_3d, generate_simplex_grid
from dal.mo_ppo import (
    MOPPONetwork,
    compute_losses,
    compute_scalar_gae,
    compute_vector_gae,
    train_moppo,
)
from dal.mo_report import build_report_markdown
from helpers import small_cfg, synthetic_data

PROTECTED_DIRS = [
    ROOT / "results",
    ROOT / "results_poa",
    ROOT / "data",
    ROOT / "data_poa",
    ROOT / "runs" / "overnight",
    ROOT / "configs",
    ROOT / "configs_poa",
]


def _snapshot_protected() -> dict[str, tuple[int, int, str]]:
    snapshot = {}
    for folder in PROTECTED_DIRS:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                st = p.stat()
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                snapshot[str(p)] = (st.st_mtime_ns, st.st_size, digest)
    return snapshot


# ============================================================================
# 1. Propriedade de Soma (§2)
# ============================================================================
def test_mo_reward_sum_property():
    cfg = small_cfg()
    data = synthetic_data()
    base_env = DroneLocEnv(data, cfg)
    mo_env = DroneLocMOEnv(base_env, memory=True)

    obs, info = mo_env.reset(seed=123, options={"w": [1.0, 1.0, 1.0, 1.0]})
    assert np.allclose(mo_env.w, [1.0, 1.0, 1.0, 1.0])

    for a in [0, 1, 3, 4, 7]:
        obs, r_scalar, term, trunc, info = mo_env.step(a)
        r_vec = info["r_vec"]
        r_vec_raw = info["r_vec_raw"]
        # With w = (1, 1, 1, 1), scalar reward MUST match vector sum exactly (tol 1e-6)
        assert np.isclose(r_scalar, float(np.sum(r_vec)), atol=1e-6)
        assert np.isclose(r_scalar, float(np.sum(r_vec_raw)), atol=1e-6)
        if term or trunc:
            break


# ============================================================================
# 2. GAE Vetorial (§3a)
# ============================================================================
def test_mo_vector_gae_equals_scalar_gaes():
    np.random.seed(42)
    T, N, M = 20, 4, 4
    rewards = np.random.randn(T, N, M).astype(np.float32)
    values = np.random.randn(T + 1, N, M).astype(np.float32)
    dones = (np.random.rand(T + 1, N) > 0.8).astype(np.float32)
    dones[0] = 0.0

    v_adv, v_ret = compute_vector_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    for j in range(M):
        s_adv, s_ret = compute_scalar_gae(
            rewards[:, :, j], values[:, :, j], dones, gamma=0.99, gae_lambda=0.95
        )
        assert np.allclose(v_adv[:, :, j].numpy(), s_adv.numpy(), atol=1e-6)
        assert np.allclose(v_ret[:, :, j].numpy(), s_ret.numpy(), atol=1e-6)


# ============================================================================
# 3. Redução para m = 1 (§3b)
# ============================================================================
def test_mo_reduction_to_scalar_ppo():
    torch.manual_seed(42)
    B, C, H, W, nv, A = 16, 7, 32, 32, 5, 8
    net_m1 = MOPPONetwork(C, H, W, nv, action_dim=A, num_objectives=1)

    obs = {"map": torch.randn(B, C, H, W), "vec": torch.randn(B, nv)}
    actions = torch.randint(0, A, (B,))
    old_logprobs = torch.randn(B)
    adv = torch.randn(B)
    returns = torch.randn(B)

    # MOPPO loss calculation
    tot_loss, pg_loss, v_loss, ent_loss = compute_losses(
        net_m1, obs, actions, old_logprobs, adv, returns, clip_range=0.2, vf_coef=0.5, ent_coef=0.01
    )

    # Standard Scalar PPO loss
    _, new_logp, entropy, val = net_m1.get_action_and_value(obs, actions)
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = torch.exp(new_logp - old_logprobs)
    pg1 = -adv_norm * ratio
    pg2 = -adv_norm * torch.clamp(ratio, 0.8, 1.2)
    expected_pg = torch.max(pg1, pg2).mean()
    expected_v = 0.5 * ((val.squeeze(-1) - returns) ** 2).mean()
    expected_ent = entropy.mean()
    expected_tot = expected_pg - 0.01 * expected_ent + 0.5 * expected_v

    assert np.isclose(tot_loss.item(), expected_tot.item(), atol=1e-6)
    assert np.isclose(pg_loss.item(), expected_pg.item(), atol=1e-6)
    assert np.isclose(v_loss.item(), expected_v.item(), atol=1e-6)
    assert np.isclose(ent_loss.item(), expected_ent.item(), atol=1e-6)


# ============================================================================
# 4. w e Histórico de Ações na Observação + Reset (§3c)
# ============================================================================
def test_mo_obs_history_and_reset():
    C, H, W = 7, 32, 32
    net = MOPPONetwork(C, H, W, 69, action_dim=8, num_objectives=4)

    obs_map = torch.randn(1, C, H, W)
    obs_v1 = torch.zeros(1, 69)
    obs_v1[0, 0] = 1.0  # w[0]

    obs_v2 = torch.zeros(1, 69)
    obs_v2[0, 3] = 1.0  # w[3]

    obs_v3 = torch.zeros(1, 69)
    obs_v3[0, 4] = 1.0  # action history slot

    val1 = net.get_value({"map": obs_map, "vec": obs_v1})
    val2 = net.get_value({"map": obs_map, "vec": obs_v2})
    val3 = net.get_value({"map": obs_map, "vec": obs_v3})

    # Critic distinguishes different weights and different history
    assert not torch.allclose(val1, val2)
    assert not torch.allclose(val1, val3)

    # Check env resets history to zeros
    cfg = small_cfg()
    data = synthetic_data()
    mo_env = DroneLocMOEnv(DroneLocEnv(data, cfg), memory=True)
    obs, _ = mo_env.reset(seed=10)
    assert np.allclose(mo_env.action_history, 0.0)
    assert obs["vec"].shape[0] == 69

    mo_env.step(3)
    assert not np.allclose(mo_env.action_history, 0.0)
    obs2, _ = mo_env.reset(seed=11)
    assert np.allclose(mo_env.action_history, 0.0)


# ============================================================================
# 5. w Fixo no Episódio e Reamostrado no Reset (§2)
# ============================================================================
def test_mo_weight_sampling_and_fixed():
    cfg = small_cfg()
    data = synthetic_data()
    mo_env = DroneLocMOEnv(DroneLocEnv(data, cfg), memory=True)

    # 1. Reset with sampled w
    obs1, info1 = mo_env.reset(seed=100)
    w1 = mo_env.w.copy()
    assert np.isclose(w1.sum(), 1.0, atol=1e-5)

    # Stays fixed during episode
    _, _, term, trunc, _ = mo_env.step(1)
    assert np.allclose(mo_env.w, w1)

    # Resampled on next reset without options
    obs2, info2 = mo_env.reset(seed=200)
    w2 = mo_env.w.copy()
    assert not np.allclose(w1, w2)

    # 2. Reset with fixed options w
    fixed_w = [0.15, 0.55, 0.15, 0.15]
    obs3, info3 = mo_env.reset(seed=300, options={"w": fixed_w})
    assert np.allclose(mo_env.w, fixed_w)
    mo_env.step(2)
    assert np.allclose(mo_env.w, fixed_w)


# ============================================================================
# 6. check_env do Gymnasium (§4)
# ============================================================================
def test_mo_check_env_passes():
    cfg = small_cfg()
    data = synthetic_data()
    base_env = DroneLocEnv(data, cfg)
    mo_env = DroneLocMOEnv(base_env, memory=True)
    check_env(mo_env, skip_render_check=True)


# ============================================================================
# 7. Hipervolume em Caso Conhecido (§6.2)
# ============================================================================
def test_mo_hypervolume_known():
    # 2D test case: non-dominated points [(1, 3), (2, 2), (3, 1)] with ref (0, 0) -> Area = 6.0
    pts_2d = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.5, 1.5]])
    ref_2d = np.array([0.0, 0.0])
    hv2 = compute_hypervolume_2d(pts_2d, ref_2d)
    assert np.isclose(hv2, 6.0, atol=1e-6)

    # 3D test case: single point (2, 3, 4) with ref (0, 0, 0) -> Volume = 24.0
    pts_3d = np.array([[2.0, 3.0, 4.0]])
    ref_3d = np.array([0.0, 0.0, 0.0])
    hv3 = compute_hypervolume_3d(pts_3d, ref_3d)
    assert np.isclose(hv3, 24.0, atol=1e-6)


# ============================================================================
# 8. Reprodutibilidade com Seed (§9)
# ============================================================================
def test_mo_reproducibility():
    cfg = small_cfg()
    data1 = synthetic_data(seed=42)
    data2 = synthetic_data(seed=42)
    env1 = DroneLocMOEnv(DroneLocEnv(data1, cfg), memory=True)
    env2 = DroneLocMOEnv(DroneLocEnv(data2, cfg), memory=True)

    obs1, _ = env1.reset(seed=42, options={"w": [0.25, 0.25, 0.25, 0.25]})
    obs2, _ = env2.reset(seed=42, options={"w": [0.25, 0.25, 0.25, 0.25]})

    assert np.allclose(obs1["map"], obs2["map"])
    assert np.allclose(obs1["vec"], obs2["vec"])

    o1, r1, t1, tr1, _ = env1.step(2)
    o2, r2, t2, tr2, _ = env2.step(2)

    assert np.isclose(r1, r2)
    assert np.allclose(o1["map"], o2["map"])
    assert np.allclose(o1["vec"], o2["vec"])
    env1.close()
    env2.close()


# ============================================================================
# 9. Colunas e Estrutura do REPORT_MO.md (§6.6 & §9)
# ============================================================================
def test_mo_report_structure(tmp_path):
    df_mo = pd.DataFrame([{
        "method": "mo_ppo", "preset": "unif", "w_str": "[0.25,0.25,0.25,0.25]",
        "w_info": 0.25, "w_cost": 0.25, "w_success": 0.25, "w_risk": 0.25,
        "success": 0.70, "success_lo": 0.65, "success_hi": 0.75,
        "false_loc": 0.15, "false_loc_lo": 0.10, "false_loc_hi": 0.20,
        "distance_m": 1200.0, "distance_lo": 1100.0, "distance_hi": 1300.0,
        "steps_median": 20.0, "decision_ms": 3.0,
    }])
    df_ref = pd.DataFrame([{
        "method": "straight", "preset": "baseline", "w_str": "N/A",
        "success": 0.70, "false_loc": 0.13, "distance_m": 2366.0,
        "steps_median": 26.0, "decision_ms": 0.005,
    }])
    hv_stats = {
        "hv": {"mo_mem": 0.5, "mo_nomem": 0.4, "fixed_ppo": 0.45},
        "eum": {"mo_mem": 0.6, "mo_nomem": 0.5, "fixed_ppo": 0.55},
        "bounds": {"success": [0.4, 0.7], "false_loc": [0.1, 0.3], "distance_m": [1000, 3000]},
    }
    h6_stats = {
        "rho_risk": -0.8, "ci_risk": [-0.9, -0.6], "p_risk": 0.001,
        "rho_cost": -0.7, "ci_cost": [-0.85, -0.5], "p_cost": 0.002,
    }

    report_file = tmp_path / "REPORT_MO.md"
    text = build_report_markdown(
        df_mo, None, df_ref, None, None, hv_stats, h6_stats, [], "Validation Passed", report_file
    )

    assert "H4" in text
    assert "H5" in text
    assert "H6" in text
    assert "H7" in text
    assert "Limitações" in text
    assert "Fronteira de Pareto" in text
    assert "Hipervolume" in text
    assert report_file.exists()


# ============================================================================
# 10. Teste de Proteção (§0 & §9)
# ============================================================================
def test_mo_protected_folders_untouched(tmp_path):
    before = _snapshot_protected()

    # Run MO steps in tmp_path
    cfg = small_cfg()
    data = synthetic_data()
    env = DroneLocMOEnv(DroneLocEnv(data, cfg), memory=True)
    obs, _ = env.reset(seed=42)
    env.step(0)
    env.close()

    after = _snapshot_protected()
    assert before == after, "Protected directories were modified during MO execution!"
