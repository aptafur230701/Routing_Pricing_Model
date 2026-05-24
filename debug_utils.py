"""
debug_utils.py
==============
Utilidades de debugging numérico para el pipeline PPO + Attention Model.
Reemplaza torch.nan_to_num() como mecanismo silencioso de corrección
por fallos inmediatos con mensajes claros.
"""

import torch


def check_tensor(name: str, x: torch.Tensor) -> None:
    """
    Verifica que un tensor no contenga NaN ni Inf.
    Falla inmediatamente con un mensaje que identifica la etapa exacta.

    Parámetros
    ----------
    name : str          — nombre descriptivo de la etapa (p.ej. "embeddings").
    x    : torch.Tensor — tensor a verificar.

    Raises
    ------
    RuntimeError si x contiene NaN o Inf.
    """
    if torch.isnan(x).any():
        raise RuntimeError(
            f"[NaN] {name} — shape {tuple(x.shape)}, "
            f"dtype {x.dtype}, device {x.device}"
        )
    if torch.isinf(x).any():
        raise RuntimeError(
            f"[Inf] {name} — shape {tuple(x.shape)}, "
            f"dtype {x.dtype}, device {x.device}"
        )
