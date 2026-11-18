"""Calibração (β₁, κ, α) com dados sintéticos e o teste de vazamento (só faixa de treino)."""
import numpy as np

from dal.v3.envs.realistic_env import valid_windows
from dal.v3.localization import calibration
from test_v3_realistic_env import _config, _tables


def test_beta1_is_higher_when_observations_match_the_map_better():
    clean, noisy = _tables(True), _tables(False)
    beta_clean = calibration.calibrate_beta1(clean, 100, window=16, n_windows=4)
    beta_noisy = calibration.calibrate_beta1(noisy, 100, window=16, n_windows=4)
    assert beta_clean["top1"] > beta_noisy["top1"]
    assert beta_clean["beta1"] >= beta_noisy["beta1"]


def test_kappa_and_alpha_are_chosen_from_their_grids():
    tables = _tables(False)
    kappa, rows = calibration.calibrate_kappa(tables, _config(), {h: 20.0 for h in (60, 100, 150)}, episodes=20)
    assert kappa in calibration.KAPPAS and len(rows) == len(calibration.KAPPAS)
    alpha, rows = calibration.calibrate_alpha(tables, _config(), episodes=20)
    assert alpha in calibration.ALPHA_GRID


def test_calibration_windows_never_leave_the_training_band():
    # Teste de vazamento: com só as linhas de treino permitidas, nenhuma janela toca as linhas de teste.
    tables = _tables(False)
    tables.rows_allowed = np.arange(0, 20)
    windows = valid_windows(tables, 16, 0.7)
    assert len(windows) > 0
    assert (windows[:, 0] + 16 <= 20).all()
