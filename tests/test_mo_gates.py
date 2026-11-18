"""Regra A4 do DECISIONS_MO_V2.md: execução com gate reprovado não pode gerar vereditos."""
import pytest
import torch

from dal.mo_eval import INVALID_GATE_MARKER, load_mo_model
from dal.mo_ppo import MOPPONetwork


def _save_dummy_model(run_dir):
    """Salva uma rede pequena com a mesma forma usada pelo load_mo_model."""
    torch.manual_seed(0)
    net = MOPPONetwork(7, 64, 64, 69, action_dim=8, num_objectives=4)
    path = run_dir / "final_model.pt"
    torch.save(net.state_dict(), path)
    return path


def test_load_refuses_invalid_gate_run(tmp_path):
    model_path = _save_dummy_model(tmp_path)
    (tmp_path / INVALID_GATE_MARKER).write_text("{}")
    with pytest.raises(ValueError, match=INVALID_GATE_MARKER):
        load_mo_model(model_path)


def test_load_allows_invalid_run_only_when_explicit(tmp_path):
    model_path = _save_dummy_model(tmp_path)
    (tmp_path / INVALID_GATE_MARKER).write_text("{}")
    assert load_mo_model(model_path, allow_invalid=True) is not None


def test_load_accepts_valid_run(tmp_path):
    assert load_mo_model(_save_dummy_model(tmp_path)) is not None
