"""
test_am_routes.py
=================
Validación rápida del pipeline AM (sin entrenamiento).

Carga los datos reales, instancia AMRoutingAgent con pesos aleatorios
y genera una ruta greedy por cada nodo de inicio.

Propósito: confirmar que encoder → contexto → decoder → ruta funciona
end-to-end con las matrices reales antes de añadir entrenamiento.
Las recompensas serán malas (pesos aleatorios), pero las rutas deben
ser ciclos válidos sin rutas -Inf.

Ejecutar:
    python test_am_routes.py
"""

import numpy as np
from config import MAX_DURATION, DEVICE
from problem_data import load_matrices, build_day_matrices
from am_agent import AMRoutingAgent

NUM_NODES = 10
TEST_DAY  = 0       # día fijo para reproducibilidad


def main():
    print("=" * 55)
    print("  Validación AM — pesos aleatorios (sin entrenar)")
    print("=" * 55)

    # ── Cargar datos reales ───────────────────────────────────
    print("\nCargando matrices...")
    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, _ = \
        load_matrices(NUM_NODES)

    _, reward_matrix_penalized = build_day_matrices(
        rate_stack[TEST_DAY], loads_stack[TEST_DAY], distance_arr, diesel_arr
    )
    print(f"Día de prueba : {TEST_DAY}")
    print(f"Max duration  : {MAX_DURATION} h")

    # ── Instanciar agente (pesos aleatorios) ─────────────────
    agent = AMRoutingAgent(
        num_nodes=NUM_NODES,
        d_h=128,
        n_heads=8,
        n_layers=3,
        d_ff=512,
        device=DEVICE,
    )
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"\nParámetros del modelo : {total_params:,}")
    print(f"Device                : {DEVICE}\n")

    # ── Generar ruta por cada nodo de inicio ─────────────────
    print(f"{'Nodo':>5} | {'Ruta':^35} | {'Reward':>8} | {'Tiempo':>7} | {'Válida':>6}")
    print("-" * 70)

    n_valid = 0
    for start in range(NUM_NODES):
        route, reward, duration = agent.generate_route(
            start_node=start,
            reward_matrix_penalized=reward_matrix_penalized,
            time_matrix=time_matrix,
            distance_arr=distance_arr,
            max_duration=MAX_DURATION,
        )
        valid = route is not None
        if valid:
            n_valid += 1
            route_str = str(route)
            print(f"{start:>5} | {route_str:^35} | {reward:>8.0f} | {duration:>6.1f}h | {'OK':>6}")
        else:
            print(f"{start:>5} | {'---':^35} | {'  -Inf':>8} | {'  ---':>7} | {'FAIL':>6}")

    print("-" * 70)
    print(f"\nRutas válidas: {n_valid}/{NUM_NODES}")
    print("\nNOTA: las recompensas son malas — el modelo no está entrenado.")
    print("      Lo importante es que no haya rutas -Inf.")


if __name__ == "__main__":
    main()
