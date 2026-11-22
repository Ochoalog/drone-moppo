"""Análise do experimento E (V46): melhor heurística no treino, E1–E3 e E5 em dados sintéticos."""
import numpy as np
import pandas as pd

from dal.v3.evaluation import experiment_e as exp_e


def _episodes(rates: dict, n=400, seeds=3, split="test", windows=40, rng_seed=0):
    """Episódios pareados: método → probabilidade de sucesso (políticas com seeds e os dois modos)."""
    rng = np.random.default_rng(rng_seed)
    window = rng.integers(windows, size=n)
    rows = []
    for method, p in rates.items():
        policy = method == "ppo"
        for seed in (range(seeds) if policy else [-1]):
            for mode in (("sampled", "argmax") if policy else ("—",)):
                success = rng.random(n) < p
                rows.append(pd.DataFrame({
                    "method": method, "mode": mode, "model_seed": seed, "episode": np.arange(n), "split": split,
                    "window_r0": window, "window_c0": 0, "success": success,
                    "false_loc": (~success) & (rng.random(n) < 0.3), "flight_time_s": rng.uniform(100, 400, n)}))
    return exp_e.with_method_key(pd.concat(rows, ignore_index=True))


def _regime(name, test_rates, train_rates, rng_seed=0):
    return exp_e.Regime(name, _episodes(test_rates, rng_seed=rng_seed),
                        _episodes(train_rates, split="train", rng_seed=rng_seed + 1))


def test_best_heuristic_is_chosen_on_train_with_false_loc_tiebreak():
    train = _episodes({"straight": 0.8, "random": 0.4, "lawnmower": 0.5}, split="train")
    assert exp_e.best_heuristic(train) == "straight"
    tie = train.copy()
    tie.loc[tie.method == "lawnmower", "success"] = tie.loc[tie.method == "straight", "success"].to_numpy()
    tie.loc[tie.method == "lawnmower", "false_loc"] = False
    assert exp_e.best_heuristic(tie) == "lawnmower"


def test_e1_supported_when_policy_clearly_beats_random_and_not_when_equal():
    strong = _regime("Canoas", {"ppo": 0.70, "random": 0.40, "straight": 0.65},
                     {"ppo": 0.7, "random": 0.4, "straight": 0.65})
    assert exp_e.classify(exp_e.advantage(strong, "random"), exp_e.E1_MIN_ADVANTAGE) == "suportada"
    weak = _regime("Canoas", {"ppo": 0.41, "random": 0.40}, {"ppo": 0.4, "random": 0.4})
    assert exp_e.classify(exp_e.advantage(weak, "random"), exp_e.E1_MIN_ADVANTAGE) != "suportada"


def test_e3_joint_bootstrap_detects_growing_advantage_and_e2_independent_difference():
    low = _regime("Taquari L0", {"ppo": 0.95, "random": 0.94, "straight": 0.99},
                  {"ppo": 0.95, "random": 0.93, "straight": 0.99}, rng_seed=1)
    high = _regime("Taquari L3", {"ppo": 0.60, "random": 0.35, "straight": 0.45},
                   {"ppo": 0.6, "random": 0.35, "straight": 0.45}, rng_seed=1)
    e3 = exp_e.difference_of_paired(high, low)
    assert e3.estimate > 0.15 and exp_e.classify(e3) == "suportada"
    e2 = exp_e.difference_of_independent(exp_e.advantage(high, high.best, seed=1),
                                         exp_e.advantage(low, low.best, seed=2))
    assert abs(e2.estimate - e3.estimate) < 1e-9 and exp_e.classify(e2) == "suportada"
    reverse = exp_e.difference_of_paired(low, high)
    assert exp_e.classify(reverse) == "contrária"


def test_method_summary_and_gate_rate():
    regime = _regime("Canoas", {"ppo": 0.7, "random": 0.4}, {"ppo": 0.7, "random": 0.4})
    summary = exp_e.method_summary(regime, "ppo@sampled")
    assert summary["n_seeds"] == 3 and summary["success_low"] <= summary["success"] <= summary["success_high"]
    assert 100 <= summary["flight_time_success_s"] <= 400
    rate = exp_e.gate_rate([{"advantage": 0.06, "threshold": 0.05, "passed": True},
                            {"advantage": 0.01, "threshold": 0.05, "passed": True},
                            {"advantage": 0.0, "threshold": 0.05, "passed": False}])
    assert rate["gate_original_passed"] == 1 and rate["gate_amended_passed"] == 2 and rate["n_runs"] == 3
