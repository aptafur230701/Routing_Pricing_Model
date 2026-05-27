"""
inference.py
============
Inferencia en tiempo real con el modelo AM entrenado.

Carga el checkpoint am_checkpoint_10nodes.pt (hasta ahora es el unico modelo que tengo) y genera una ruta óptima
desde el nodo que el usuario ingrese, usando los datos del día más reciente.

Uso
---
    python inference.py
"""

import os
import torch
import numpy as np

from config import DEVICE, MAX_DURATION
from am_agent import AMRoutingAgent
from problem_data import load_matrices, build_day_matrices

# ── Configuración ─────────────────────────────────────────────────────────────

NUM_NODES       = 10
CHECKPOINT_NAME = f"am_checkpoint_{NUM_NODES}nodes.pt"


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_agent(checkpoint_path: str) -> AMRoutingAgent:
    """Instancia el agente y carga los pesos del checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint: {checkpoint_path}\n"
            f"Asegúrate de haber ejecutado main.py antes de correr inference.py."
        )

    agent = AMRoutingAgent(num_nodes=NUM_NODES, device=DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_latest_day():
    """Carga las matrices y toma el último día disponible como snapshot actual."""
    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, _ = \
        load_matrices(NUM_NODES)

    # Último día del stack histórico = día más reciente disponible
    _, reward_matrix_penalized = build_day_matrices(
        rate_stack[-1],
        loads_stack[-1],
        distance_arr,
        diesel_arr,
    )

    return time_matrix, reward_matrix_penalized, distance_arr


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(agent: AMRoutingAgent, start_node: int,
                  time_matrix, reward_matrix_penalized, distance_arr):
    """Genera una ruta greedy desde start_node y muestra el resultado."""
    import time

    print(f"\n  Generando ruta desde nodo {start_node}...")

    t0 = time.perf_counter()
    route, total_reward, time_elapsed = agent.generate_route(
        start_node=start_node,
        reward_matrix_penalized=reward_matrix_penalized,
        time_matrix=time_matrix,
        distance_arr=distance_arr,
        max_duration=MAX_DURATION,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 50)
    if route is None:
        print("  ⚠  No se encontró una ruta válida desde ese nodo.")
        print(f"     (Tiempo máximo permitido: {MAX_DURATION:.1f} h)")
    else:
        ruta_str = " → ".join(str(n) for n in route)
        print(f"  Ruta      : {ruta_str}")
        print(f"  Reward    : {total_reward:.0f}")
        print(f"  Duración  : {time_elapsed:.2f} h  (máx {MAX_DURATION:.1f} h)")
        print(f"  Nodos vis.: {len(route) - 1} de {NUM_NODES}")
    print(f"  Inferencia: {inference_ms:.2f} ms")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cwd             = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(cwd, CHECKPOINT_NAME)

    print("=" * 50)
    print(f"  AM Inference — {NUM_NODES} nodos")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {CHECKPOINT_NAME}")
    print("=" * 50)

    # Cargar modelo y datos una sola vez
    print("\nCargando modelo...")
    agent = load_agent(checkpoint_path)
    print("Cargando datos del día más reciente...")
    time_matrix, reward_matrix_penalized, distance_arr = load_latest_day()
    print("Listo.\n")

    # Loop interactivo: el usuario elige el nodo de inicio
    while True:
        print(f"Ingresa el nodo de inicio (0 a {NUM_NODES - 1}), o 'q' para salir:")
        entrada = input("  > ").strip()

        if entrada.lower() == "q":
            print("\nSaliendo. ¡Hasta luego!")
            break

        if not entrada.isdigit():
            print(f"  ✗ Entrada inválida. Ingresa un número entre 0 y {NUM_NODES - 1}.\n")
            continue

        start_node = int(entrada)
        if start_node < 0 or start_node >= NUM_NODES:
            print(f"  ✗ Nodo fuera de rango. Debe estar entre 0 y {NUM_NODES - 1}.\n")
            continue

        run_inference(agent, start_node, time_matrix, reward_matrix_penalized, distance_arr)
        print()


if __name__ == "__main__":
    main()
