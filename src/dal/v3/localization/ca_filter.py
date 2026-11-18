"""Verossimilhança ciente da mudança (CA, change-aware), §3.4 do plano v3.

L_x ∝ (1 − ρ_x)·exp(β·s_x) + ρ_x·ū,  com ρ_x = α·m_x  e  ū = média de exp(β·s) nas células válidas da janela.

m_x ∈ [0, 1] é o prior de mudança da célula (fração inundada segundo uma fonte independente da imagem
observada). Onde o mapa provavelmente mudou, a célula passa a "acreditar" menos na comparação
mapa × observação e se aproxima da verossimilhança média (não informativa). Com α = 0, a verossimilhança
é idêntica à padrão.
"""
from __future__ import annotations

import torch

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def change_aware_likelihood(likelihood: torch.Tensor, change_prior: torch.Tensor, alpha: float,
                            valid: torch.Tensor | None = None) -> torch.Tensor:
    """Aplica a mistura CA a verossimilhanças (N, ...) já exponenciadas.

    `likelihood` pode estar em qualquer escala comum por episódio (por exemplo, normalizada pelo máximo):
    a mistura é homogênea, então a escala não muda a crença posterior.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha precisa estar em [0, 1]")
    if alpha == 0.0:
        return likelihood
    n = likelihood.shape[0]
    flat = likelihood.reshape(n, -1)
    mask = torch.ones_like(flat, dtype=torch.bool) if valid is None else valid.reshape(n, -1)
    mean_valid = (flat * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
    rho = alpha * change_prior.reshape(n, -1).to(flat.dtype).clamp(0.0, 1.0)
    return ((1.0 - rho) * flat + rho * mean_valid).reshape(likelihood.shape)
