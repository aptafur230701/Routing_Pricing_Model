"""
critic_head.py
==============
CriticHead: estima el valor del estado V(s_t) desde el vector de contexto h_t.

Usado en el loop PPO de am_training.py para calcular la ventaja:
    A_t = R_t - V(s_t)

Entrada : h_t  (batch, d_h)  — vector de contexto de ContextNetwork.
Salida  : V(s) (batch,)      — escalar por elemento del batch.

Compatibilidad
--------------
· Sin dependencias de agent.py, training.py ni routing_env.py.
· d_h debe coincidir con el d_h de AttentionEncoder y ContextNetwork.
"""

import torch
import torch.nn as nn


class CriticHead(nn.Module):
    """
    MLP de dos capas sobre el vector de contexto h_t → V(s).

    Arquitectura: d_h → d_h//2 → ReLU → 1 (lineal)

    El output NO tiene activación final para permitir valores negativos
    (recompensas negativas son posibles en el problema de ruteo).

    Parámetros
    ----------
    d_h : int — dimensión del embedding (igual que AttentionEncoder).
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Linear(d_h // 2, 1),
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        h_t : Tensor (batch, d_h)

        Retorna
        -------
        Tensor (batch,) — estimación de V(s) por elemento.
        """
        return self.net(h_t).squeeze(-1)
