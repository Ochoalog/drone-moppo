"""Estatística da v3: diferenças pareadas, rliable e vereditos com dados sintéticos de efeito conhecido."""
import numpy as np
import pandas as pd

from dal.v3.evaluation import verdicts


def _episodes(effects: dict[str, tuple[float, float, float]], seeds=range(10), n=300, seed=0):
    """effects[método] = (p_sucesso, p_falsa, tempo médio); heurísticas usam model_seed = -1."""
    rng = np.random.default_rng(seed)
    windows = rng.integers(0, 60, size=(n, 2))
    # Dificuldade comum por episódio (como nos dados reais pareados): o mesmo sorteio para todos os métodos.
    difficulty, false_draw = rng.random(n), rng.random(n)
    rows = []
    for method, (p_success, p_false, time_s) in effects.items():
        for model_seed in ([-1] if not method.startswith("ppo") else list(seeds)):
            jitter = 0.004 * (model_seed - 4.5) if model_seed >= 0 else 0.0
            success = difficulty < p_success + jitter
            false = ~success & (false_draw < p_false / max(1e-9, 1 - p_success))
            rows.append(pd.DataFrame({"method": method, "model_seed": model_seed, "region": "taquari",
                                      "scenario": "S2", "episode": np.arange(n), "window_r0": windows[:, 0],
                                      "window_c0": windows[:, 1], "success": success, "false_loc": false,
                                      "flight_time_s": rng.normal(time_s, 20, n)}))
    return pd.concat(rows, ignore_index=True)


def test_paired_difference_recovers_a_large_effect():
    episodes = _episodes({"ppo_ca": (0.85, 0.05, 150.0), "straight_ca": (0.60, 0.15, 220.0)})
    diff = verdicts.paired_difference(episodes, "ppo_ca", "straight_ca", "success")
    assert diff.low > 0.15 and diff.high < 0.40


def test_h3_superiority_and_efficiency_paths():
    clear = _episodes({"ppo_ca": (0.85, 0.05, 150.0), "straight_ca": (0.60, 0.15, 220.0)})
    result = verdicts.verdict_h3_for_place(clear, "ppo_ca", "straight_ca")
    assert result["supported"] and result["superiority"]
    tie_faster = _episodes({"ppo_ca": (0.62, 0.10, 120.0), "straight_ca": (0.62, 0.10, 220.0)})
    result = verdicts.verdict_h3_for_place(tie_faster, "ppo_ca", "straight_ca")
    assert result["non_inferior_efficient"] and not result["superiority"]


def test_h2_detects_lower_false_localization_with_ca():
    episodes = _episodes({"straight": (0.60, 0.20, 200.0), "straight_ca": (0.62, 0.08, 200.0),
                          "ppo_ca_stdfilter": (0.70, 0.20, 150.0), "ppo_ca": (0.71, 0.08, 150.0)})
    result = verdicts.verdict_h2({"taquari_S2": episodes, "poa_ilhas": episodes})
    assert result["verdict"] == "suportada"


def test_iqm_and_holm():
    scores = {"a": np.linspace(0.5, 0.9, 10).reshape(10, 1), "b": np.full((1, 1), 0.6)}
    table = verdicts.iqm_with_ci(scores, reps=200)
    assert table.set_index("method").loc["a", "ci_low"] <= 0.7 <= table.set_index("method").loc["a", "ci_high"]
    adjusted = verdicts.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adjusted["x"] == 0.03 and adjusted["y"] == 0.06 and adjusted["z"] == 0.06


def test_h1_query_cell_bootstrap_detects_a_paired_gain():
    from dal.v3.evaluation.verdicts import query_cell_difference, verdict_h1

    rng = np.random.default_rng(1)
    n = 4000
    base = {"window": np.zeros(n, int), "variant": np.tile([0, 1], n // 2), "cell": np.repeat(np.arange(n // 2), 2)}
    hand = dict(base, hit=rng.random(n) < 0.3)
    better = dict(base, hit=hand["hit"] | (rng.random(n) < 0.15))
    same = dict(base, hit=hand["hit"].copy())
    gain = query_cell_difference(better, hand, reps=300)
    assert gain.low > 0
    tie = query_cell_difference(same, hand, reps=300)
    assert tie.low <= 0 <= tie.high
    assert verdict_h1({"S1": gain, "S2": gain})["verdict"] == "suportada"
    assert verdict_h1({"S1": gain, "S2": tie})["verdict"] == "inconclusiva"
