"""
transfer_learning.py
====================
Transfer learning estandarizado para la cadena de escalado:

    10 → 20 → 35 → 50 → 75 → 97 nodos

Por qué el transfer es siempre 100% en este modelo
---------------------------------------------------
Ninguna capa del AMRoutingAgent ni del CriticHead tiene dimensiones
que dependan de num_nodes. El Transformer opera sobre conjuntos de
embeddings de tamaño variable — lo que cambia entre tamaños es el
número de nodos en runtime, no los pesos del modelo.

  encoder.input_proj   : 5 → d_h          (independiente de N)
  encoder.layers[i]    : d_h → d_h        (independiente de N)
  context_net.proj     : 2·d_h+3 → d_h   (independiente de N)
  decoder.glimpse      : d_h → d_h        (independiente de N)
  decoder.pointer      : d_h → logit      (independiente de N)
  critic.net           : d_h → 1          (independiente de N)

Uso desde main.py
-----------------
  from transfer_learning import transfer_checkpoint, NODE_SEQUENCE

  # Obtener checkpoint del tamaño anterior automáticamente
  source = get_source_checkpoint(cwd, num_nodes=20)
  agent, critic = transfer_checkpoint(source, target_num_nodes=20)

Uso standalone
--------------
  python transfer_learning.py
  → Verifica que la cadena completa de transfers es válida.
"""

import os
import torch
from am_agent import AMRoutingAgent
from critic_head import CriticHead
from config import DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Secuencia canónica de escalado
# ─────────────────────────────────────────────────────────────────────────────

NODE_SEQUENCE = [10, 20, 35, 50, 75, 97]


def get_source_checkpoint(base_dir: str, target_num_nodes: int) -> str | None:
    """
    Dado el tamaño destino, devuelve la ruta al checkpoint del tamaño
    anterior en NODE_SEQUENCE, o None si target es el primero (10 nodos).

    Busca primero en results_{N}nodes/ y luego en la raíz del proyecto.

    Parámetros
    ----------
    base_dir         : directorio raíz del proyecto.
    target_num_nodes : tamaño al que vas a entrenar.

    Retorna
    -------
    str con la ruta al .pt fuente, o None si no hay predecesor.
    """
    if target_num_nodes not in NODE_SEQUENCE:
        raise ValueError(
            f"{target_num_nodes} no está en NODE_SEQUENCE {NODE_SEQUENCE}.\n"
            f"Añádelo si quieres usarlo en la cadena de transfer."
        )

    idx = NODE_SEQUENCE.index(target_num_nodes)

    # Primer tamaño de la cadena — no tiene predecesor
    if idx == 0:
        return None

    source_nodes = NODE_SEQUENCE[idx - 1]
    fname = f"am_checkpoint_{source_nodes}nodes.pt"

    # Buscar: 1) carpeta compartida checkpoints/  2) results_Nnodes/  3) raíz
    candidates = [
        os.path.join(base_dir, fname),
        os.path.join(os.path.dirname(base_dir), "checkpoints", fname),
        os.path.join(base_dir, f"results_{source_nodes}nodes", fname),
        os.path.join(base_dir, fname),
    ]
    # eliminar duplicados manteniendo orden
    seen, candidates = set(), [
        p for p in candidates if not (p in seen or seen.add(p))
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No se encontró el checkpoint de {source_nodes} nodos en ninguna de:\n"
        + "\n".join(f"  · {p}" for p in candidates)
        + f"\nCopia el archivo a checkpoints/ antes de entrenar {target_num_nodes} nodos."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transfer principal
# ─────────────────────────────────────────────────────────────────────────────

def transfer_checkpoint(
    source_checkpoint: str,
    target_num_nodes:  int,
    d_h:      int = 128,
    n_heads:  int = 8,
    n_layers: int = 3,
    d_ff:     int = 512,
    verbose:  bool = True,
) -> tuple:
    """
    Transfiere los pesos de un checkpoint entrenado a un agente de mayor tamaño.

    Parámetros
    ----------
    source_checkpoint : ruta al .pt del modelo fuente.
    target_num_nodes  : tamaño del grafo destino.
    d_h, n_heads, n_layers, d_ff : hiperparámetros (deben coincidir con la fuente).
    verbose           : imprime resumen del transfer.

    Retorna
    -------
    agent  : AMRoutingAgent inicializado con pesos transferidos.
    critic : CriticHead     inicializado con pesos transferidos.
    """
    if not os.path.exists(source_checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {source_checkpoint}")

    checkpoint = torch.load(source_checkpoint, map_location=DEVICE)

    # Instanciar modelos destino
    agent  = AMRoutingAgent(
        num_nodes=target_num_nodes,
        d_h=d_h, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        device=DEVICE,
    )
    critic = CriticHead(d_h).to(DEVICE)

    # Transferir pesos — filtrar por forma por seguridad
    agent_sd  = agent.state_dict()
    critic_sd = critic.state_dict()

    transferred, skipped = [], []

    for k, v in checkpoint["agent"].items():
        if k in agent_sd and v.shape == agent_sd[k].shape:
            agent_sd[k] = v
            transferred.append(k)
        else:
            skipped.append(k)

    for k, v in checkpoint["critic"].items():
        if k in critic_sd and v.shape == critic_sd[k].shape:
            critic_sd[k] = v
            transferred.append(k)
        else:
            skipped.append(k)

    agent.load_state_dict(agent_sd)
    critic.load_state_dict(critic_sd)

    total = len(agent_sd) + len(critic_sd)
    pct   = 100 * len(transferred) / total

    if verbose:
        source_name = os.path.basename(source_checkpoint)
        print(f"\n  [Transfer] {source_name} → {target_num_nodes} nodos")
        print(f"             {len(transferred)}/{total} tensores ({pct:.1f}%)", end="")
        if skipped:
            print(f" — omitidos: {skipped}")
        else:
            print(" ✓")

    return agent, critic


# ─────────────────────────────────────────────────────────────────────────────
# Integración con main.py
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_for_training(
    base_dir:        str,
    num_nodes:       int,
    use_transfer:    bool = True,
    d_h:      int = 128,
    n_heads:  int = 8,
    n_layers: int = 3,
    d_ff:     int = 512,
    verbose:  bool = True,
) -> tuple:
    """
    Función de conveniencia para main.py.

    Si use_transfer=True y existe un checkpoint del tamaño anterior en
    NODE_SEQUENCE, inicializa el agente con esos pesos (transfer learning).
    Si no, crea un agente con pesos aleatorios.

    Parámetros
    ----------
    base_dir     : directorio raíz del proyecto (donde están results_Nnodes/).
    num_nodes    : tamaño del grafo a entrenar.
    use_transfer : activar/desactivar transfer learning.

    Retorna
    -------
    agent  : AMRoutingAgent listo para entrenar.
    critic : CriticHead listo para entrenar.
    """
    if use_transfer and num_nodes in NODE_SEQUENCE:
        try:
            source = get_source_checkpoint(base_dir, num_nodes)
            if source is not None:
                return transfer_checkpoint(
                    source, num_nodes, d_h, n_heads, n_layers, d_ff, verbose
                )
        except FileNotFoundError as e:
            print(f"\n  ⚠ Transfer omitido: {e}")
            print(f"    Iniciando {num_nodes} nodos con pesos aleatorios.\n")

    # Sin transfer: pesos aleatorios
    if verbose:
        print(f"\n  [Transfer] {num_nodes} nodos — pesos aleatorios (sin fuente).")
    agent  = AMRoutingAgent(num_nodes, d_h, n_heads, n_layers, d_ff, device=DEVICE)
    critic = CriticHead(d_h).to(DEVICE)
    return agent, critic


# ─────────────────────────────────────────────────────────────────────────────
# Verificación de la cadena completa (standalone)
# ─────────────────────────────────────────────────────────────────────────────

def verify_chain(base_dir: str):
    """
    Verifica qué checkpoints existen y cuáles transfers son posibles
    en la cadena completa NODE_SEQUENCE.
    """
    print("\n" + "=" * 55)
    print("  Transfer Learning — Estado de la cadena")
    print(f"  Secuencia: {' → '.join(str(n) for n in NODE_SEQUENCE)}")
    print("=" * 55)

    for i, n in enumerate(NODE_SEQUENCE):
        # Buscar checkpoint de este tamaño
        candidates = [
            os.path.join(base_dir, f"results_{n}nodes", f"am_checkpoint_{n}nodes.pt"),
            os.path.join(base_dir, f"am_checkpoint_{n}nodes.pt"),
        ]
        exists = next((p for p in candidates if os.path.exists(p)), None)

        if i == 0:
            origen = "base"
        else:
            prev = NODE_SEQUENCE[i - 1]
            origen = f"← {prev} nodos"

        estado = f"✓  {exists}" if exists else "✗  (no entrenado aún)"
        print(f"  {n:>3} nodos  [{origen:>12}]  {estado}")

    print("=" * 55 + "\n")


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    verify_chain(cwd)

    # Demo: simular transfer de 10 → 20 si existe el checkpoint
    try:
        source = get_source_checkpoint(cwd, target_num_nodes=20)
        agent, critic = transfer_checkpoint(source, target_num_nodes=20, verbose=True)
        total_params = sum(p.numel() for p in agent.parameters())
        print(f"  Parámetros transferidos al agente de 20 nodos: {total_params:,}")
    except FileNotFoundError as e:
        print(f"  Demo omitida: {e}")


if __name__ == "__main__":
    main()
