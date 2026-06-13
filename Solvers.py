import numpy as np
from collections import namedtuple

# ── Constantes de dominio (antes números mágicos dispersos por el archivo) ────
DAYS_PER_PERIOD = 14            # día de mercado:  day_idx = start + int(t // 14)
TIME_EPS = 1e-6                 # tolerancia de holgura temporal
INSERTION_TIME_PENALTY = 1e-3  # peso del castigo por tiempo en la inserción LNS
_BERNOULLI_SEED_A = 9973       # constantes de semilla del MC rollout
_BERNOULLI_SEED_B = 97

# torch y pulp se importan de forma PEREZOSA dentro de las únicas funciones que
# los usan (generate_optimal_route_pytorch y solve_lp_relaxation), de modo que
# importar Solvers no arrastre dependencias pesadas/opcionales.

# ── Tipos de resultado (namedtuple = 100% compatible con desempaquetado por
#    tupla, así que los llamadores existentes siguen funcionando sin cambios) ──
RHResult = namedtuple("RHResult", "status route reward duration is_valid")
MetaResult = namedtuple("MetaResult", "status route reward duration")
RewardResult = namedtuple("RewardResult", "reward duration")


def _day_index(start_day_idx, time_elapsed, max_day):
    """Índice de día de mercado para un tiempo transcurrido (lógica única
    replicada antes en cada solver)."""
    return min(start_day_idx + int(time_elapsed // DAYS_PER_PERIOD), max_day)


def _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                   start_day_idx, max_day, as_numpy=True):
    """Devuelve un closure get_rm(t) que construye y cachea (por llamada) la
    matriz de reward del día correspondiente a `t`.

    as_numpy=True  → np.ndarray (acceso rm[i, j]); usado por la familia lookahead,
                     stochastic y el rollout.
    as_numpy=False → pd.DataFrame (acceso rm[i][j] / rm.iloc[i, j]); usado por
                     solve_heuristic_rolling_horizon(_stochastic) y
                     simulate_route_reward, que dependen de ese tipo/indexación.

    Centraliza el patrón `_get_rm` que estaba duplicado ~6 veces. NO cambia qué
    matriz ve cada llamador ni cómo la indexa: solo unifica construcción y caché.
    """
    from problem_data import build_day_matrices
    cache = {}

    def get_rm(time_elapsed):
        day_idx = _day_index(start_day_idx, time_elapsed, max_day)
        if day_idx not in cache:
            _, rm_pen = build_day_matrices(
                rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
            )
            cache[day_idx] = np.array(rm_pen, dtype=float) if as_numpy else rm_pen
        return cache[day_idx]

    return get_rm


def _close_cycle_and_finalize(route, current_node, time_elapsed, start_node,
                              time_m, max_d, start_day_idx,
                              rate_stack, loads_stack, distance_arr, diesel_arr,
                              avail_prob_arr):
    """Cierra el ciclo (regreso al depósito si es factible en tiempo), valida y
    re-evalúa el reward de forma canónica con simulate_route_reward.

    Captura el bloque-cola idéntico que estaba copiado en las 5 funciones de la
    familia rolling-horizon + MC rollout. `time_m` ya es np.ndarray en todas, por
    lo que time_m[a, b] reproduce exactamente el acceso previo (sea [a][b] o [a,b]).

    Devuelve un RHResult (status, route, reward, duration, is_valid).
    """
    if route[-1] != start_node:
        return_time = float(time_m[current_node, start_node])
        if time_elapsed + return_time <= max_d:
            time_elapsed += return_time
            route.append(start_node)

    if route[-1] != start_node or len(route) < 2:
        return RHResult("Infeasible", route if len(route) > 1 else None,
                        -np.inf, time_elapsed, False)

    total_reward, total_duration = simulate_route_reward(
        route, start_node, start_day_idx,
        time_m, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=avail_prob_arr,
    )
    is_valid = total_duration <= max_d
    status = "Optimal" if is_valid else "Infeasible"
    return RHResult(status, route, total_reward, total_duration, is_valid)


def generate_optimal_route_pytorch(agent, start_node, time_matrix, reward_matrix,
                                   NUM_NODES, MAX_DURATION, MAX_STEPS_PER_EPISODE):
    """Genera una ruta con la política aprendida (selección greedy), evitando
    revisitar nodos intermedios. Si se alcanza max_steps intenta el regreso
    forzado si es factible, y valida la ventana de duración final.

    Devuelve (route, total_reward, time_elapsed) si forma un ciclo válido, o
    (None, -inf, inf) si no logra cerrar el ciclo.
    """
    import torch  # import perezoso: única función del módulo que usa torch

    max_steps = MAX_STEPS_PER_EPISODE
    agent.epsilon = 0
    agent.policy_net.eval()
    current_node = start_node
    time_elapsed = 0.0
    state = np.array([current_node, time_elapsed / MAX_DURATION], dtype=np.float32)
    route = [start_node]
    visited_intermediate_nodes = set()  # nodos visitados distintos de start_node
    total_reward = 0.0
    returned_home = False

    try:
        with torch.no_grad():
            for step in range(max_steps):
                # --- Selección de acción ---
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                q_values_numpy = q_values.cpu().data.numpy()[0]

                # --- Enmascarar acciones inválidas ---
                q_values_numpy[current_node] = -np.inf  # no quedarse en el mismo nodo
                for visited_node_idx in visited_intermediate_nodes:
                    if 0 <= visited_node_idx < len(q_values_numpy):
                        q_values_numpy[visited_node_idx] = -np.inf

                # --- Mejor acción válida ---
                next_node = np.argmax(q_values_numpy)

                # ¿existe alguna acción válida?
                if q_values_numpy[next_node] == -np.inf:
                    # Sin movimientos válidos: intentar regreso forzado al depósito.
                    if current_node != start_node:
                        return_time = time_matrix[current_node][start_node]
                        if time_elapsed + return_time <= MAX_DURATION + TIME_EPS:
                            next_node = start_node  # forzar regreso
                        else:
                            returned_home = False
                            break  # no se puede continuar
                    else:  # atascado en el depósito (no debería pasar con el masking)
                        returned_home = False
                        break

                # --- Simular paso ---
                step_time = time_matrix[current_node][next_node]
                step_reward = reward_matrix[current_node][next_node]

                # --- Violación inmediata de tiempo ---
                if time_elapsed + step_time > MAX_DURATION + TIME_EPS and next_node != start_node:
                    returned_home = False
                    break

                # --- Actualizar estado ---
                time_elapsed += step_time
                total_reward += step_reward
                current_node = next_node
                route.append(current_node)
                if current_node != start_node:
                    visited_intermediate_nodes.add(current_node)

                state = np.array([current_node, min(time_elapsed, MAX_DURATION) / MAX_DURATION],
                                 dtype=np.float32)

                # --- Regreso natural ---
                if current_node == start_node:
                    returned_home = True
                    break

            # --- Regreso forzado si se agotaron los pasos ---
            if not returned_home and current_node != start_node:
                return_time = time_matrix[current_node][start_node]
                return_reward = reward_matrix[current_node][start_node]
                if time_elapsed + return_time <= MAX_DURATION + TIME_EPS:
                    time_elapsed += return_time
                    total_reward += return_reward
                    current_node = start_node
                    route.append(start_node)
                    returned_home = True
    finally:
        # Garantiza que la red vuelva a modo entrenamiento incluso si algo falla
        # a mitad (el original solo lo hacía en la ruta feliz).
        agent.policy_net.train()

    # --- Validación final ---
    is_cycle = (returned_home and route[0] == start_node
                and route[-1] == start_node and len(route) > 1)
    if not is_cycle:
        return None, -np.inf, np.inf  # ruta fallida

    intermediate_nodes = route[1:-1]
    has_duplicates = len(intermediate_nodes) != len(set(intermediate_nodes))
    if has_duplicates:
        print(f"Warning: DRL route {route} has duplicate intermediate nodes despite masking!")

    # En ambos casos (válido o con duración/nodos inválidos) se devuelven los
    # detalles del ciclo, igual que en el original.
    return route, total_reward, time_elapsed

def solve_heuristic(start_node, time_m, reward_m, max_d, num_n):
    current_node = start_node
    time_elapsed = 0.0
    total_reward = 0.0
    route = [start_node]
    visited = {start_node}
    steps = 0
    max_arcs = num_n - 1

    while steps < max_arcs:
        best_reward = -np.inf  
        best_next_node = None

        for next_node in range(num_n):
            if next_node != current_node and next_node not in visited:
                step_time = time_m[current_node][next_node]
                return_time = time_m[next_node][start_node]
                total_future_time = time_elapsed + step_time + return_time
                if total_future_time <= max_d:          # ← filtro de tiempo primero
                    step_reward = reward_m[current_node][next_node]
                    return_reward = reward_m[next_node][start_node]
                    total_cycle_reward = step_reward + return_reward
                    if total_cycle_reward > best_reward:
                        best_reward = total_cycle_reward
                        best_next_node = next_node

        if best_next_node is not None:
            time_elapsed += time_m[current_node][best_next_node]
            total_reward += reward_m[current_node][best_next_node]
            current_node = best_next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break  # CAMBIO: sin return_threshold, sale directo cuando no hay arcos positivos

    # Cerrar ciclo
    if route[-1] != start_node:
        return_time = time_m[current_node][start_node]
        if time_elapsed + return_time <= max_d:
            time_elapsed += return_time
            total_reward += reward_m[current_node][start_node]
            route.append(start_node)

    is_valid = (route[-1] == start_node and len(route) > 1
                and time_elapsed <= max_d)
    status = "Optimal" if is_valid else "Infeasible"
    return RHResult(status, route, total_reward, time_elapsed, is_valid)


def solve_lp_relaxation(start_node, time_m, reward_m, max_d, num_n):
    """Relajación LP de la variante VRP para obtener una cota superior.

    Nota: se relajan deliberadamente las restricciones de eliminación de subtours
    (no se añaden cortes tipo MTZ); para una *cota superior* esto sigue siendo
    válido. Por eso ya no se declaran las variables `u` ni la lista `other_nodes`,
    que estaban muertas en la versión original.
    """
    import pulp  # import perezoso: única función del módulo que usa pulp

    nodes = list(range(num_n))

    # Modelo LP
    lp_prob = pulp.LpProblem(f"VRP_LP_Relaxation_{start_node}", pulp.LpMaximize)

    # Variables de decisión (continuas en [0, 1])
    x = pulp.LpVariable.dicts("Route", (nodes, nodes), 0, 1, pulp.LpContinuous)

    # Mismo objetivo y restricciones que el MIP
    lp_prob += pulp.lpSum(reward_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    for k in nodes:
        lp_prob += pulp.lpSum(x[k][j] for j in nodes if k != j) == pulp.lpSum(x[j][k] for j in nodes if k != j)
        if k == start_node:
            lp_prob += pulp.lpSum(x[start_node][j] for j in nodes if j != start_node) == 1
            lp_prob += pulp.lpSum(x[j][start_node] for j in nodes if j != start_node) == 1
        else:
            lp_prob += pulp.lpSum(x[j][k] for j in nodes if j != k) <= 1

    total_time = pulp.lpSum(time_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)
    lp_prob += total_time <= max_d

    # Solve LP
    solver = pulp.PULP_CBC_CMD(msg=0)
    lp_prob.solve(solver)

    status = pulp.LpStatus[lp_prob.status]
    upper_bound = pulp.value(lp_prob.objective) if status == 'Optimal' else np.inf

    return status, upper_bound

def _node_insertion_pass(route, time_m, reward_m, max_d, num_n):
    """
    Intenta insertar nodos no visitados en la mejor posición de la ruta.
    Acepta la inserción solo si mejora el reward total y es factible en tiempo.
    Repite hasta que ninguna inserción mejore.
    """
    max_arcs = num_n - 1
    improved = True
    best_route = route[:]
    best_reward = sum(reward_m[best_route[k]][best_route[k+1]]
                      for k in range(len(best_route) - 1))
    best_time = sum(time_m[best_route[k]][best_route[k+1]]
                    for k in range(len(best_route) - 1))

    while improved:
        improved = False
        visited = set(best_route)
        unvisited = [n for n in range(num_n) if n not in visited]

        for node in unvisited:
            if len(best_route) - 1 >= max_arcs:
                break  # ruta ya en el límite de arcos

            best_gain = 0  # solo aceptar si hay ganancia neta positiva
            best_pos = None

            for pos in range(1, len(best_route)):
                a = best_route[pos - 1]
                b = best_route[pos]

                delta_time = (time_m[a][node] + time_m[node][b]
                              - time_m[a][b])
                delta_reward = (reward_m[a][node] + reward_m[node][b]
                                - reward_m[a][b])

                if best_time + delta_time <= max_d and delta_reward > best_gain:
                    best_gain = delta_reward
                    best_pos = pos

            if best_pos is not None:
                best_route.insert(best_pos, node)
                best_reward += best_gain
                best_time = sum(time_m[best_route[k]][best_route[k+1]]
                                for k in range(len(best_route) - 1))
                improved = True
                break  # reiniciar con la ruta actualizada

    return best_route, best_reward, best_time


def solve_2opt_heuristic(start_node, time_m, reward_m, max_d, num_n):
    status, route, total_reward, total_time, is_valid = solve_heuristic(
        start_node, time_m, reward_m, max_d, num_n
    )
    if not is_valid:
        return RHResult(status, route, total_reward, total_time, is_valid)

    best_route = route[:]
    best_reward = total_reward
    best_time = total_time
    outer_improved = True

    while outer_improved:
        outer_improved = False

        # Fase 1: 2-opt swaps
        swap_improved = True
        while swap_improved:
            swap_improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = (best_route[:i]
                                 + best_route[i:j+1][::-1]
                                 + best_route[j+1:])
                    new_time = sum(time_m[new_route[k]][new_route[k+1]]
                                   for k in range(len(new_route) - 1))
                    new_reward = sum(reward_m[new_route[k]][new_route[k+1]]
                                     for k in range(len(new_route) - 1))
                    if (new_time <= max_d
                            and new_route[0] == start_node
                            and new_route[-1] == start_node
                            and new_reward > best_reward):
                        best_route = new_route
                        best_reward = new_reward
                        best_time = new_time
                        swap_improved = True
                        outer_improved = True
                        break
                if swap_improved:
                    break

        # Fase 2: inserción de nodos no visitados
        inserted_route, inserted_reward, inserted_time = _node_insertion_pass(
            best_route, time_m, reward_m, max_d, num_n
        )
        if inserted_reward > best_reward:
            best_route = inserted_route
            best_reward = inserted_reward
            best_time = inserted_time
            outer_improved = True

    is_valid = (best_time <= max_d
                and best_route[0] == start_node
                and best_route[-1] == start_node)
    status = "Optimal" if is_valid else "Infeasible"
    return RHResult(status, best_route, best_reward, best_time, is_valid)


def solve_LNS_metaheuristic(start_node, time_m, reward_m, max_d, num_n, seed=None):
    """
    Large Neighborhood Search (LNS) metaheuristic:
    1. Start with a greedy initial solution
    2. Iteratively destroy and repair neighborhoods
    3. Accept solutions if they improve best known or pass probabilistic criterion
    Returns: status, route, total_reward, total_duration (same format as solve_mip)
    """
    # Aislamiento del RNG: guardamos el estado global y lo restauramos antes de
    # cada return, para no contaminar el RNG global del proceso. Seguimos usando
    # np.random.seed(seed) para que la SECUENCIA de números sea idéntica a la del
    # original (no se cambia a default_rng, que produciría otra secuencia).
    _rng_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    max_arcs = num_n - 1

    # --- Step 1: Generate initial solution using greedy heuristic ---
    status, init_route, init_reward, init_time, is_valid = solve_heuristic(
        start_node, time_m, reward_m, max_d, num_n
    )

    # Bug 3 fix: heuristic may reject all arcs (e.g. all rewards negative) and
    # return an invalid seed. Fall back to the nearest feasible out-and-back.
    if not is_valid or init_route == [start_node]:
        fallback_route = None
        fallback_reward = -np.inf
        fallback_time = np.inf
        for neighbor in range(num_n):
            if neighbor == start_node:
                continue
            t = time_m[start_node][neighbor] + time_m[neighbor][start_node]
            if t <= max_d:
                r = reward_m[start_node][neighbor] + reward_m[neighbor][start_node]
                if fallback_route is None or r > fallback_reward:
                    fallback_route = [start_node, neighbor, start_node]
                    fallback_reward = r
                    fallback_time = t
        if fallback_route is None:
            if _rng_state is not None:
                np.random.set_state(_rng_state)
            return MetaResult("Infeasible", None, -np.inf, np.inf)
        init_route = fallback_route
        init_reward = fallback_reward
        init_time = fallback_time

    best_route = init_route[:]
    best_reward = init_reward
    best_time = init_time
    current_route = init_route[:]
    current_reward = init_reward
    current_time = init_time

    # --- LNS Parameters ---
    max_iterations = max(50, num_n * num_n)
    neighborhood_size = max(2, min(4, len(best_route) - 2))  # Size of segment to destroy
    temperature = best_reward * 0.1  # For simulated annealing acceptance
    cooling_rate = 0.95
    patience = 20
    no_improve_count = 0

    # --- Step 2: LNS Main Loop ---
    for iteration in range(max_iterations):

        # --- Destroy Phase: Remove a neighborhood (segment) ---
        if len(current_route) > 3:
            max_start = len(current_route) - neighborhood_size - 1
            if max_start > 1:
                destroy_start = np.random.randint(1, max_start)
                destroy_end = min(destroy_start + neighborhood_size, len(current_route) - 1)
                destroyed_route = current_route[:destroy_start] + current_route[destroy_end:]
            else:
                # Route too short to safely destroy, skip this iteration
                destroyed_route = current_route[:]
        else:
            destroyed_route = current_route[:]

        # --- Repair Phase: Reinsert removed nodes optimally ---
        removed_segment = current_route[destroy_start:destroy_end] if len(current_route) > 3 and max_start > 1 else []
        repaired_route, repaired_reward, repaired_time = _repair_route_lns_best_position(
            destroyed_route, removed_segment,
            start_node, time_m, reward_m, max_d
        )

        # --- Evaluate repaired solution ---
        if repaired_route is not None:
            # Check feasibility
            is_feasible = (repaired_time <= max_d and
                          repaired_route[0] == start_node and
                          repaired_route[-1] == start_node and
                          len(repaired_route) - 1 <= max_arcs)  # Max steps constraint

            if is_feasible:
                # --- Acceptance Criterion: Simulated Annealing ---
                delta_reward = repaired_reward - current_reward

                if delta_reward > 0:
                    # Accept improving solution
                    current_route = repaired_route
                    current_reward = repaired_reward
                    current_time = repaired_time
                    no_improve_count = 0

                    # Update best known solution
                    if current_reward > best_reward:
                        best_route = current_route[:]
                        best_reward = current_reward
                        best_time = current_time

                else:
                    # Accept worse solution with probability (diversification)
                    acceptance_prob = np.exp(delta_reward / max(temperature, 1e-6))
                    if np.random.rand() < acceptance_prob:
                        current_route = repaired_route
                        current_reward = repaired_reward
                        current_time = repaired_time
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # --- Cooling and Early Stopping ---
        temperature *= cooling_rate

        if no_improve_count >= patience:
            break

    # --- Final Validation ---
    is_valid = (best_route is not None and
                best_route[0] == start_node and
                best_route[-1] == start_node and
                best_time <= max_d and
                len(best_route) - 1 <= max_arcs)

    final_status = "Optimal" if is_valid else "Infeasible"

    if _rng_state is not None:
        np.random.set_state(_rng_state)
    return MetaResult(final_status, best_route, best_reward, best_time)


def _repair_route_lns_best_position(destroyed_route, removed_nodes, start_node, time_m, reward_m, max_d):
    """
    Improved Best-Position Insertion for LNS (edge-based reward version)

    - Uses local delta-time and delta-reward updates
    - Evaluates true benefit of each insertion position
    - Avoids O(n^2) recomputation overhead
    - Inserts nodes in best global order
    """

    # --- If no nodes to insert, just validate route ---
    if not removed_nodes:
        if destroyed_route[-1] != start_node:
            destroyed_route.append(start_node)

        total_time = sum(time_m[destroyed_route[i]][destroyed_route[i+1]]
                         for i in range(len(destroyed_route)-1))
        total_reward = sum(reward_m[destroyed_route[i]][destroyed_route[i+1]]
                           for i in range(len(destroyed_route)-1))

        if total_time <= max_d:
            return destroyed_route, total_reward, total_time
        else:
            return None, -np.inf, np.inf

    # Work on a copy
    current_route = destroyed_route[:]

    # Precompute initial time & reward
    current_time = sum(time_m[current_route[i]][current_route[i+1]]
                       for i in range(len(current_route)-1))
    current_reward = sum(reward_m[current_route[i]][current_route[i+1]]
                         for i in range(len(current_route)-1))

    remaining = removed_nodes[:]

    # --- Insert nodes one-by-one, always picking highest-benefit insertion first ---
    while remaining:

        best_global_gain = -np.inf
        best_node = None
        best_pos = None
        best_new_time = None
        best_new_reward = None

        # Evaluate each removed node
        for v in remaining:

            # Try every insertion position except index 0 (start node)
            for pos in range(1, len(current_route)):

                a = current_route[pos - 1]     # predecessor
                b = current_route[pos]         # successor

                # Δ time
                old_t = time_m[a][b]
                new_t = time_m[a][v] + time_m[v][b]
                delta_t = new_t - old_t

                # Δ reward
                old_r = reward_m[a][b]
                new_r = reward_m[a][v] + reward_m[v][b]
                delta_r = new_r - old_r

                # If insertion violates max duration → skip
                new_time = current_time + delta_t
                if new_time > max_d:
                    continue

                # Scoring function: reward benefit - penalty * time increase
                # (tunable weighting)
                score = delta_r - INSERTION_TIME_PENALTY * max(delta_t, 0)

                if score > best_global_gain:
                    best_global_gain = score
                    best_node = v
                    best_pos = pos
                    best_new_time = new_time
                    best_new_reward = current_reward + delta_r

        # If no feasible insertion → fail
        if best_node is None:
            return None, -np.inf, np.inf

        # --- Perform the best insertion ---
        current_route.insert(best_pos, best_node)
        remaining.remove(best_node)
        current_time = best_new_time
        current_reward = best_new_reward

    # After all insertions, ensure route closes properly
    if current_route[-1] != start_node:
        a = current_route[-1]
        b = start_node

        current_time += time_m[a][b]
        current_reward += reward_m[a][b]

        if current_time > max_d:
            return None, -np.inf, np.inf

        current_route.append(start_node)

    return current_route, current_reward, current_time

def solve_genetic_algorithm(start_node, time_m, reward_m, max_d, num_n, seed=None):
    """
    Hybrid Genetic Algorithm for routing:
    1. Initialize population with diverse feasible routes (max 6 nodes: 5 steps)
    2. Apply crossover, mutation (2-opt), and local search operators
    3. Return best solution from population
    Returns: status, route, total_reward, total_duration (same format as solve_mip)
    """
    # Aislamiento del RNG global (ver nota en solve_LNS_metaheuristic).
    _rng_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    max_route_len = num_n

    # --- GA Parameters (scaled by problem size) ---
    population_size = max(10, num_n * num_n)
    num_generations = max(30, num_n * 3)
    mutation_rate = 0.5
    crossover_rate = 0.5
    elite_size = max(2, population_size // 5)

    # --- Step 1: Initialize Population with Diverse Solutions ---
    population = []

    # Seed 1: Greedy heuristic
    status, route, reward, duration, is_valid = solve_heuristic(start_node, time_m, reward_m, max_d, num_n)
    if is_valid and len(route) <= max_route_len:
        population.append({'route': route, 'reward': reward, 'duration': duration})
    # Seed 2: Nearest neighbor with randomization
    for _ in range(max(2, population_size // 4)):
        route = _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= max_route_len:
            duration = sum(time_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            reward = sum(reward_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            if duration <= max_d:
                population.append({'route': route, 'reward': reward, 'duration': duration})
    # Seed 3: Random feasible routes
    for _ in range(max(2, population_size // 4)):
        route = _generate_random_feasible_route(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= max_route_len:
            duration = sum(time_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            reward = sum(reward_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            population.append({'route': route, 'reward': reward, 'duration': duration})
    # Fill remaining population slots
    while len(population) < population_size:
        route = _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= max_route_len:
            duration = sum(time_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            reward = sum(reward_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            if duration <= max_d:
                population.append({'route': route, 'reward': reward, 'duration': duration})
    if not population:
        if _rng_state is not None:
            np.random.set_state(_rng_state)
        return MetaResult("Infeasible", None, -np.inf, np.inf)

    best_overall = max(population, key=lambda x: x['reward'])
    gens = 0
    # --- Step 2: GA Main Loop (Generations) ---
    for generation in range(num_generations):
        # --- Selection: Keep elite + tournament selection ---
        population.sort(key=lambda x: x['reward'], reverse=True)
        elite = population[:elite_size]

        # Tournament selection for remaining population
        new_population = elite[:]
        while len(new_population) < population_size:
            # Select 3 random individuals, pick best
            tournament = [population[np.random.randint(0, len(population))] for _ in range(3)]
            winner = max(tournament, key=lambda x: x['reward'])
            new_population.append(winner)
        # --- Crossover and Mutation ---
        offspring = elite[:]  # Keep elite
        while len(offspring) < population_size:
            if np.random.rand() < crossover_rate:
                # Crossover: Combine two routes
                parent1 = new_population[np.random.randint(0, len(new_population))]
                parent2 = new_population[np.random.randint(0, len(new_population))]

                child_route = _crossover_routes(
                    parent1['route'], parent2['route'], start_node, time_m, reward_m, max_d, num_n
                )

                if child_route and len(child_route) <= max_route_len:
                    # Apply mutation (2-opt improvement)
                    if np.random.rand() < mutation_rate:
                        child_route = _apply_2opt_mutation(child_route, start_node, time_m, reward_m, max_d, num_n)

                    # Evaluate child
                    child_duration = sum(time_m[child_route[i]][child_route[i+1]] for i in range(len(child_route) - 1))
                    child_reward = sum(reward_m[child_route[i]][child_route[i+1]] for i in range(len(child_route) - 1))

                    if child_duration <= max_d and len(child_route) <= max_route_len:
                        offspring.append({'route': child_route, 'reward': child_reward, 'duration': child_duration})
            else:
                # Pure mutation (2-opt on existing solution)
                parent = new_population[np.random.randint(0, len(new_population))]
                mutant_route = _apply_2opt_mutation(parent['route'][:], start_node, time_m, reward_m, max_d, num_n)

                if mutant_route and len(mutant_route) <= max_route_len:
                    mutant_duration = sum(time_m[mutant_route[i]][mutant_route[i+1]] for i in range(len(mutant_route) - 1))
                    mutant_reward = sum(reward_m[mutant_route[i]][mutant_route[i+1]] for i in range(len(mutant_route) - 1))

                    if mutant_duration <= max_d and len(mutant_route) <= max_route_len:
                        offspring.append({'route': mutant_route, 'reward': mutant_reward, 'duration': mutant_duration})
        # --- Replace population (keep best from offspring) ---
        population = offspring[:population_size]

        # --- Track best solution ---
        current_best = max(population, key=lambda x: x['reward'])
        if current_best['reward'] > best_overall['reward']:
            best_overall = current_best
        gens +=1
    # --- Final Validation ---
    is_valid = (best_overall is not None and
                best_overall['route'][0] == start_node and
                best_overall['route'][-1] == start_node and
                best_overall['duration'] <= max_d and
                len(best_overall['route']) <= max_route_len and
                len(best_overall['route']) >= 2)

    final_status = "Optimal" if is_valid else "Infeasible"

    if _rng_state is not None:
        np.random.set_state(_rng_state)
    return MetaResult(final_status, best_overall['route'],
                      best_overall['reward'], best_overall['duration'])


def _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n):
    """
    Nearest neighbor heuristic with randomization:
    Start from a node and greedily move to closest unvisited node.
    Max 5 steps (6 nodes total including start/end)
    """
    max_steps = num_n - 1
    max_route_len = num_n
    current_node = start_node
    route = [start_node]
    visited = {start_node}
    time_elapsed = 0.0
    steps = 0

    while steps < max_steps:
        # Find nearest unvisited node that keeps us feasible
        best_next = None
        best_distance = np.inf

        for next_node in range(num_n):
            if next_node not in visited:
                distance = time_m[current_node][next_node]
                return_distance = time_m[next_node][start_node]
                if time_elapsed + distance + return_distance <= max_d and distance < best_distance:
                    best_distance = distance
                    best_next = next_node
        if best_next is not None:
            time_elapsed += time_m[current_node][best_next]
            current_node = best_next
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break

    if len(route) == max_route_len and current_node != start_node:
        # Force return home if we have max_steps steps but not yet home
        #Change the last step to return home
        node_to_delete = route[-1]
        route = route[0:-1]
        current_node = route[-1]
        time_to_delete = time_m[current_node][node_to_delete]
        time_elapsed = time_elapsed - time_to_delete
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time
    elif len(route) < max_route_len and route[-1] != start_node:
        current_node = route[-1]
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time

    if time_elapsed <= max_d and route[0] == route[-1] == start_node and len(route) <= max_route_len:
        return route
    return None


def _generate_random_feasible_route(start_node, time_m, reward_m, max_d, num_n):
    """
    Generate a random feasible route by randomly selecting nodes.
    Max 5 steps (6 nodes total including start/end)
    """
    max_steps = num_n - 1
    max_route_len = num_n
    current_node = start_node
    route = [start_node]
    visited = {start_node}
    time_elapsed = 0.0
    steps = 0

    while steps < max_steps:
        # Get all unvisited nodes that keep us feasible
        candidates = []
        for next_node in range(num_n):
            if next_node not in visited:
                distance = time_m[current_node][next_node]
                return_distance = time_m[next_node][start_node]

                if time_elapsed + distance + return_distance <= max_d:
                    candidates.append(next_node)

        if candidates:
            next_node = candidates[np.random.randint(0, len(candidates))]
            time_elapsed += time_m[current_node][next_node]
            current_node = next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break

    # Return home
    if len(route) == max_route_len and current_node != start_node:
        # Force return home if we have max_steps steps but not yet home
        #Change the last step to return home
        node_to_delete = route[-1]
        route = route[0:-1]
        current_node = route[-1]
        time_to_delete = time_m[current_node][node_to_delete]
        time_elapsed = time_elapsed - time_to_delete
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time
    elif len(route) < max_route_len and route[-1] != start_node:
        current_node = route[-1]
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time

    if time_elapsed <= max_d and route[0] == route[-1] == start_node and len(route) <= max_route_len:
        return route
    return None


def _crossover_routes(route1, route2, start_node, time_m, reward_m, max_d, num_n):
    """
    Order Crossover (OX): Combines two routes by:
    1. Copy segment from parent1
    2. Fill remaining nodes from parent2 in order
    Respects max num_n nodes constraint
    """
    max_route_len = num_n
    if len(route1) < 4 or len(route2) < 4:
        return route1 if np.random.rand() < 0.5 else route2

    # Extract intermediate nodes (exclude start/end)
    # (nodes1/nodes2 eran variables muertas: se eliminaron.)

    # Select random segment from parent1
    seg_start = np.random.randint(1, min(len(route1) - 2, 4))  # Keep segment reasonable
    seg_end = np.random.randint(seg_start + 1, len(route1) - 1)

    # Initialize child with segment from parent1
    child = [start_node] + route1[seg_start:seg_end]
    child_nodes = set(child[1:])

    # Fill remaining nodes from parent2 in order, respecting max length
    for node in route2[1:-1]:
        if node not in child_nodes and len(child) < max_route_len - 1:  # Leave room for return to start
            child.append(node)
            child_nodes.add(node)

    # Close the route
    child.append(start_node)

    # Validate feasibility
    if len(child) > max_route_len:
        child = child[:max_route_len]
        child[-1] = start_node

    duration = sum(time_m[child[i]][child[i+1]] for i in range(len(child) - 1))

    if duration <= max_d and len(child) <= max_route_len and len(child) >= 2:
        return child
    return None


def _apply_2opt_mutation(route, start_node, time_m, reward_m, max_d, num_n):
    """
    2-opt mutation: Try reversing a segment to improve the route.
    Maintains max num_n nodes constraint.
    """
    max_route_len = num_n
    best_route = route[:]
    best_reward = sum(reward_m[route[i]][route[i+1]] for i in range(len(route) - 1))
    improved = True
    iterations = 0
    max_iterations = 10

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                # Reverse segment
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]

                # Skip if exceeds max length
                if len(new_route) > max_route_len:
                    continue

                new_duration = sum(time_m[new_route[k]][new_route[k+1]] for k in range(len(new_route) - 1))
                new_reward = sum(reward_m[new_route[k]][new_route[k+1]] for k in range(len(new_route) - 1))

                if new_duration <= max_d and new_reward > best_reward:
                    best_route = new_route
                    best_reward = new_reward
                    improved = True
                    break
            if improved:
                break

    return best_route if len(best_route) <= max_route_len else None


def _apply_lns_mutation(route, start_node, time_m, reward_m, max_d, destroy_size, num_n):
    """
    LNS mutation operator used by HGA-LNS:
    1. Destroy: remove `destroy_size` random intermediate nodes
    2. Repair:  reinsert using best-position greedy strategy
    3. Refine:  polish geometry with 2-opt
    Falls back to the original route if repair produces nothing valid.
    """
    max_route_len = num_n
    intermediates = route[1:-1]
    if len(intermediates) < 2:
        return route  # too short to destroy meaningfully

    k = min(destroy_size, len(intermediates) - 1)  # keep at least 1 intermediate node
    removed_idx_set = set(np.random.choice(len(intermediates), k, replace=False).tolist())
    removed_nodes = [intermediates[i] for i in removed_idx_set]
    destroyed_route = (
        [start_node]
        + [n for i, n in enumerate(intermediates) if i not in removed_idx_set]
        + [start_node]
    )

    repaired_route, _, repaired_time = _repair_route_lns_best_position(
        destroyed_route, removed_nodes, start_node, time_m, reward_m, max_d
    )

    if repaired_route is None or repaired_time > max_d or len(repaired_route) > max_route_len:
        return route  # fallback to original

    refined = _apply_2opt_mutation(repaired_route, start_node, time_m, reward_m, max_d, num_n)
    if refined and len(refined) <= max_route_len:
        return refined
    return repaired_route


def solve_HGA_LNS_metaheuristic(
    start_node, time_m, max_d, num_n,
    rate_stack, loads_stack, distance_arr, diesel_arr, start_day_idx,
    seed=None,
):
    """
    Hybrid Genetic Algorithm – Large Neighborhood Search (HGA-LNS):
    · GA drives global exploration: diverse population, tournament selection, OX crossover.
    · LNS destroy/repair replaces classical mutation for targeted local exploitation.
    · 2-opt refinement polishes each offspring's geometry after repair.

    Internal fitness uses simulate_route_reward() with sequential arrival days
    (avail_prob_arr=None, deterministic world), so every individual in the
    population is evaluated on the same footing as DRL Det.
    Search operators (crossover, LNS repair, 2-opt) still use the static day-0
    reward matrix for fast local decisions.

    Returns: status, route, total_reward, total_duration  (same format as solve_mip)
    """
    from problem_data import build_day_matrices
    _, reward_m = build_day_matrices(
        rate_stack[start_day_idx], loads_stack[start_day_idx], distance_arr, diesel_arr
    )
    time_m_np = np.array(time_m, dtype=float)

    # Caché de matrices de reward (numpy) por día, vía el helper compartido.
    # Semánticamente idéntico a simulate_route_reward(..., avail_prob_arr=None):
    # construye la matriz del día bajo demanda y la cachea durante esta llamada.
    max_day = rate_stack.shape[0] - 1
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, max_day, as_numpy=True)

    # Aislamiento del RNG global (ver nota en solve_LNS_metaheuristic).
    _rng_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    max_route_len = num_n

    # --- Parameters ---
    population_size = max(10, num_n * num_n)
    num_generations  = max(30, num_n * 3)
    crossover_rate   = 0.6
    lns_mut_rate     = 0.7   # probability of applying LNS mutation to each offspring
    elite_size       = max(2, population_size // 5)
    destroy_size     = max(1, min(2, num_n - 3))  # nodes removed per destroy op

    # --- Step 1: Initialize Population (identical seeding strategy to GA) ---
    population = []

    def _eval(route):
        """Fast canonical fitness: matrices numpy cacheadas, días secuenciales."""
        if route is None or len(route) < 2:
            return -np.inf, np.inf
        t = 0.0
        r = 0.0
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            r += _get_rm(t)[i, j]
            t += time_m_np[i, j]
        return r, t

    _, route, _, _, is_valid = solve_heuristic(
        start_node, time_m, reward_m, max_d, num_n
    )
    if is_valid and len(route) <= max_route_len:
        rew, dur = _eval(route)
        population.append({'route': route, 'reward': rew, 'duration': dur})

    for _ in range(max(2, population_size // 4)):
        route = _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= max_route_len:
            rew, dur = _eval(route)
            if dur <= max_d:
                population.append({'route': route, 'reward': rew, 'duration': dur})

    for _ in range(max(2, population_size // 4)):
        route = _generate_random_feasible_route(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= max_route_len:
            rew, dur = _eval(route)
            if dur <= max_d:
                population.append({'route': route, 'reward': rew, 'duration': dur})

    _fill_iters = 0
    while len(population) < population_size and _fill_iters < population_size * 10:
        _fill_iters += 1
        route = _generate_random_feasible_route(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= max_route_len:
            rew, dur = _eval(route)
            if dur <= max_d:
                population.append({'route': route, 'reward': rew, 'duration': dur})

    # Sweep garantizado: todas las rutas de 2 paradas (s→a→s) y 3 paradas (s→a→b→s).
    # Se ejecuta SIEMPRE, no solo cuando la población está vacía.
    # Garantiza que rutas cortas óptimas estén en la población aunque el greedy
    # estático (reward_m del start_day_idx) las descarte por preferir otro orden.
    # Evaluadas con _eval (días secuenciales reales) para consistencia con _eval.
    for _a in range(num_n):
        if _a == start_node:
            continue
        _t2 = float(time_m_np[start_node, _a] + time_m_np[_a, start_node])
        if _t2 <= max_d:
            _r2, _d2 = _eval([start_node, _a, start_node])
            population.append({'route': [start_node, _a, start_node], 'reward': _r2, 'duration': _d2})
        for _b in range(num_n):
            if _b == start_node or _b == _a:
                continue
            _t3 = float(time_m_np[start_node, _a] + time_m_np[_a, _b] + time_m_np[_b, start_node])
            if _t3 <= max_d:
                _r3, _d3 = _eval([start_node, _a, _b, start_node])
                population.append({'route': [start_node, _a, _b, start_node], 'reward': _r3, 'duration': _d3})

    if not population:
        return "Infeasible", None, -np.inf, np.inf

    best_overall = max(population, key=lambda x: x['reward'])

    # --- Step 2: HGA-LNS Main Loop ---
    for _ in range(num_generations):
        population.sort(key=lambda x: x['reward'], reverse=True)
        elite = population[:elite_size]

        # Tournament selection to fill mating pool
        mating_pool = elite[:]
        while len(mating_pool) < population_size:
            tournament = [population[np.random.randint(0, len(population))] for _ in range(3)]
            mating_pool.append(max(tournament, key=lambda x: x['reward']))

        # Crossover + LNS-mutation to build offspring
        offspring = elite[:]
        _off_iters = 0
        while len(offspring) < population_size:
            _off_iters += 1
            if _off_iters > population_size * 30:
                break
            if np.random.rand() < crossover_rate:
                p1 = mating_pool[np.random.randint(0, len(mating_pool))]
                p2 = mating_pool[np.random.randint(0, len(mating_pool))]
                child_route = _crossover_routes(
                    p1['route'], p2['route'], start_node, time_m, reward_m, max_d, num_n
                )
            else:
                # No crossover: clone a tournament winner to mutate
                child_route = mating_pool[np.random.randint(0, len(mating_pool))]['route'][:]

            if not child_route or len(child_route) > max_route_len:
                continue

            # LNS mutation: destroy → repair → 2-opt (replaces classical mutation)
            if np.random.rand() < lns_mut_rate:
                child_route = _apply_lns_mutation(
                    child_route, start_node, time_m, reward_m, max_d, destroy_size, num_n
                )

            if child_route and len(child_route) <= max_route_len:
                child_rew, child_dur = _eval(child_route)
                # Tolerancia negativa: evita que el drift de acumulación de floats
                # en _eval deje pasar rutas marginalmente infeasibles (e.g. 77.63 > 77.0).
                if child_dur <= max_d - 1e-6:
                    offspring.append({'route': child_route, 'reward': child_rew, 'duration': child_dur})

        population = offspring[:population_size]

        current_best = max(population, key=lambda x: x['reward'])
        if current_best['reward'] > best_overall['reward']:
            best_overall = current_best

    # --- Final Validation ---
    is_valid = (
        best_overall['route'][0] == start_node
        and best_overall['route'][-1] == start_node
        and best_overall['duration'] <= max_d
        and 2 <= len(best_overall['route']) <= max_route_len
    )
    final_status = "Optimal" if is_valid else "Infeasible"
    if not is_valid:
        # best_overall tiene duration > max_d por drift de float en _eval.
        # Fallback: barremos todas las rutas de 2 y 3 paradas y buscamos la mejor
        # que sea feasible según simulate_route_reward (bit-exact con el resto).
        fallback_best_route  = None
        fallback_best_reward = -np.inf
        for _a in range(num_n):
            if _a == start_node:
                continue
            for candidate in ([start_node, _a, start_node],):
                _t = float(time_m_np[start_node, _a] + time_m_np[_a, start_node])
                if _t > max_d:
                    continue
                _r, _d = simulate_route_reward(
                    candidate, start_node, start_day_idx,
                    time_m_np, rate_stack, loads_stack, distance_arr, diesel_arr,
                    avail_prob_arr=None,
                )
                if _d <= max_d and _r > fallback_best_reward:
                    fallback_best_reward = _r
                    fallback_best_route  = candidate
            for _b in range(num_n):
                if _b == start_node or _b == _a:
                    continue
                _t = float(time_m_np[start_node, _a] + time_m_np[_a, _b] + time_m_np[_b, start_node])
                if _t > max_d:
                    continue
                candidate = [start_node, _a, _b, start_node]
                _r, _d = simulate_route_reward(
                    candidate, start_node, start_day_idx,
                    time_m_np, rate_stack, loads_stack, distance_arr, diesel_arr,
                    avail_prob_arr=None,
                )
                if _d <= max_d and _r > fallback_best_reward:
                    fallback_best_reward = _r
                    fallback_best_route  = candidate
        if fallback_best_route is not None:
            if _rng_state is not None:
                np.random.set_state(_rng_state)
            return MetaResult(
                "Optimal", fallback_best_route, fallback_best_reward,
                float(sum(time_m_np[fallback_best_route[i], fallback_best_route[i+1]]
                          for i in range(len(fallback_best_route)-1))))
        if _rng_state is not None:
            np.random.set_state(_rng_state)
        return MetaResult(final_status, None, -np.inf, np.inf)

    # Canonical re-evaluation via simulate_route_reward guarantees bit-exact
    # parity with all other solvers. The internal _eval uses pre-cached numpy
    # arrays (_rm_cache) whose float accumulation can diverge slightly from
    # simulate_route_reward. Passing avail_prob_arr=None keeps this deterministic.
    total_reward, total_duration = simulate_route_reward(
        best_overall['route'], start_node, start_day_idx,
        time_m_np, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    if _rng_state is not None:
        np.random.set_state(_rng_state)
    return MetaResult(final_status, best_overall['route'], total_reward, total_duration)


def simulate_route_reward(
    route, start_node, start_day_idx,
    time_matrix_np, rate_stack, loads_stack,
    distance_arr, diesel_arr,
    avail_prob_arr=None,
):
    """Compute the actual reward for a fixed route using sequential arrival days
    and the same Bernoulli lane-availability draws as beam_search_dynamic.

    Replicates beam_search_dynamic exactly:
    - day_idx = min(start_day_idx + int(time_elapsed // 14), max_day)
    - draw_lane_availability(start_day_idx, node=i, arrival_day=day_idx, ...)
      applied on every arc except those departing from or arriving at start_node,
      matching the skip-condition on line "current_node != start_node" in
      beam_search_dynamic.
    - If a lane is blocked (lane_exists[j] == 0), that arc yields 0 reward
      (no cargo to haul; this is consistent with the BIG_M_PENALTY the MIP
      assigns to blocked arcs to avoid them during planning).

    If avail_prob_arr is None, no Bernoulli filtering is applied (backward
    compatible with callers that do not supply availability data).
    """
    from problem_data import draw_lane_availability

    if route is None or len(route) < 2:
        return RewardResult(-np.inf, np.inf)

    max_day   = rate_stack.shape[0] - 1
    num_nodes = distance_arr.shape[0]
    # Caché compartido en modo DataFrame: simulate_route_reward indexa con
    # rm.iloc[i, j], por lo que necesita el DataFrame (as_numpy=False).
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, max_day, as_numpy=False)

    time_elapsed = 0.0
    total_reward = 0.0

    for step in range(len(route) - 1):
        i       = route[step]
        j       = route[step + 1]
        day_idx = _day_index(start_day_idx, time_elapsed, max_day)
        rm      = _get_rm(time_elapsed)
        arc_r   = float(rm.iloc[i, j])

        # Mirror beam_search_dynamic: skip Bernoulli when departing from or
        # arriving at start_node (those arcs are always available).
        if (avail_prob_arr is not None
                and i != start_node
                and j != start_node):
            lane_exists = draw_lane_availability(
                start_day_idx, node=i, arrival_day=day_idx,
                avail_prob_arr=avail_prob_arr, num_nodes=num_nodes,
            )
            if lane_exists[j] == 0:
                arc_r = 0.0

        total_reward += arc_r
        time_elapsed += float(time_matrix_np[i][j])

    return RewardResult(total_reward, time_elapsed)


def solve_heuristic_rolling_horizon(
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
):
    """Greedy miope con día de mercado DINÁMICO (rolling horizon).

    Baseline no-aprendido honesto: ve EXACTAMENTE la misma información que la
    política DRL (día corriente según time_elapsed), pero decide arco-a-arco sin
    razonamiento de horizonte. Recalcula la matriz de reward del día corriente en
    cada paso, replicando _get_current_day_idx() del entorno:
        day_idx = min(start_day_idx + int(time_elapsed // 14), num_days - 1)

    Devuelve: status, route, total_reward, total_duration, is_valid
    (mismo formato que solve_heuristic).
    """
    num_days = rate_stack.shape[0]
    # DataFrame (as_numpy=False): esta función indexa rm[current_node][next_node]
    # (acceso por columnas de pandas), por lo que debe conservar ese tipo.
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, num_days - 1, as_numpy=False)

    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    current_node = start_node
    time_elapsed = 0.0
    route = [start_node]
    visited = {start_node}
    steps = 0
    max_arcs = num_n - 1

    while steps < max_arcs:
        # Decision uses the current-day matrix (causal, same info as DRL).
        rm = _get_rm(time_elapsed)

        best_score = -np.inf
        best_next_node = None

        for next_node in range(num_n):
            if next_node != current_node and next_node not in visited:
                step_time = time_m[current_node][next_node]
                return_time = time_m[next_node][start_node]
                if time_elapsed + step_time + return_time <= max_d:
                    score = rm[current_node][next_node] + rm[next_node][start_node]
                    if score > best_score:
                        best_score = score
                        best_next_node = next_node

        if best_next_node is not None:
            time_elapsed += time_m[current_node][best_next_node]
            current_node = best_next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break

    # Cierre de ciclo + validación + evaluación canónica (helper compartido).
    return _close_cycle_and_finalize(
        route, current_node, time_elapsed, start_node, time_m, max_d,
        start_day_idx, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )


def solve_heuristic_rolling_horizon_lookahead(
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    lookahead: int = 3,
):
    """Rolling Horizon Greedy with Deterministic Lookahead.

    Extiende solve_heuristic_rolling_horizon reemplazando el scoring miope de
    1 paso por una simulación greedy recursiva de `lookahead` pasos hacia
    adelante.  En cada paso de decisión, para cada candidato `next` se simula
    la mejor ruta greedy de hasta `lookahead-1` pasos adicionales desde `next`
    (usando los días de mercado correctos en cada sub-paso) y se acumula el
    reward total de esa sub-ruta como score.

    Causalidad: los sub-pasos del lookahead usan la misma lógica de día
    dinámico que el solver padre (day_idx = start_day_idx + int(t // 14)),
    por lo que no hay información del futuro — solo planificación hacia adelante
    con información del presente.

    Complejidad: O(N^(lookahead+1)) por paso de decisión en el peor caso,
    pero poda agresiva por factibilidad temporal la hace tratable para N<=100
    y lookahead<=4.

    Parámetros
    ----------
    lookahead : int — pasos de simulación hacia adelante por candidato (default 3).
                lookahead=1 es equivalente a solve_heuristic_rolling_horizon.

    Devuelve: status, route, total_reward, total_duration, is_valid
    (mismo formato que solve_heuristic_rolling_horizon).
    """
    num_days  = rate_stack.shape[0]
    max_arcs  = num_n - 1

    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    # Cache de matrices de reward por día (numpy: esta función indexa rm[a, b]).
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, num_days - 1, as_numpy=True)

    # ── Función de lookahead recursiva ────────────────────────────────────────
    def _lookahead_score(node, t_elapsed, visited_set, depth):
        """
        Retorna el mejor reward acumulado greedy de `depth` pasos desde `node`,
        más el reward del retorno final al start_node.

        En cada nivel elige el candidato que maximiza el reward inmediato más
        el lookahead recursivo restante.  Si ningún candidato es factible,
        retorna solo el reward del retorno directo desde `node`.
        """
        rm = _get_rm(t_elapsed)

        # Retorno directo desde este nodo (siempre disponible como fallback)
        return_reward = rm[node, start_node]

        if depth == 0:
            return return_reward

        best_extended = -np.inf

        for cand in range(num_n):
            if cand == node or cand in visited_set:
                continue
            step_time   = time_m[node, cand]
            return_time = time_m[cand, start_node]
            # Factibilidad: el candidato debe permitir regresar al depot
            if t_elapsed + step_time + return_time > max_d:
                continue

            arc_reward    = rm[node, cand]
            t_after       = t_elapsed + step_time
            visited_after = visited_set | {cand}

            # Reward acumulado: arco actual + mejor lookahead desde el candidato
            candidate_score = arc_reward + _lookahead_score(
                cand, t_after, visited_after, depth - 1
            )

            if candidate_score > best_extended:
                best_extended = candidate_score

        # Si encontramos al menos un candidato factible, usamos el score extendido;
        # de lo contrario caemos al retorno directo.
        return best_extended if best_extended > -np.inf else return_reward

    # ── Loop principal ────────────────────────────────────────────────────────
    current_node  = start_node
    time_elapsed  = 0.0
    route         = [start_node]
    visited       = {start_node}
    steps         = 0

    while steps < max_arcs:
        rm = _get_rm(time_elapsed)

        best_score     = -np.inf
        best_next_node = None

        for next_node in range(num_n):
            if next_node == current_node or next_node in visited:
                continue
            step_time   = time_m[current_node, next_node]
            return_time = time_m[next_node, start_node]
            if time_elapsed + step_time + return_time > max_d:
                continue

            arc_reward = rm[current_node, next_node]
            t_after    = time_elapsed + step_time
            # Score = reward inmediato + mejor lookahead de (lookahead-1) pasos
            score = arc_reward + _lookahead_score(
                next_node, t_after, visited | {next_node}, lookahead - 1
            )

            if score > best_score:
                best_score     = score
                best_next_node = next_node

        if best_next_node is not None:
            time_elapsed += time_m[current_node, best_next_node]
            current_node  = best_next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break

    # Cierre de ciclo + validación + evaluación canónica (helper compartido).
    return _close_cycle_and_finalize(
        route, current_node, time_elapsed, start_node, time_m, max_d,
        start_day_idx, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )


def solve_heuristic_rolling_horizon_lookahead_stochastic(
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    avail_prob_arr,
    lookahead=3,
):
    """Extensión estocástica de solve_heuristic_rolling_horizon_lookahead.

    Reemplaza el scoring determinista de cada sub-paso del lookahead por sorteos
    Bernoulli con las mismas semillas que el resto del sistema: en cada nivel de
    recursión se llama a draw_lane_availability con seed basada en
    (start_day_idx, node, arrival_day), exactamente igual que
    solve_heuristic_rolling_horizon_stochastic y RoutingEnv.

    La evaluación final usa simulate_route_reward(..., avail_prob_arr=avail_prob_arr)
    para garantizar paridad bit-exact con DRL Real y RH-Greedy Real.

    Con lookahead=1 la función es idéntica a solve_heuristic_rolling_horizon_stochastic.
    """
    from problem_data import draw_lane_availability

    num_days = rate_stack.shape[0]
    max_day  = num_days - 1
    max_arcs = num_n - 1

    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    # Cache de matrices de reward por día (numpy: indexa rm[a, b]).
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, max_day, as_numpy=True)

    def _lookahead_score(node, t_elapsed, visited_set, depth):
        """
        Mejor reward acumulado greedy (con filtrado Bernoulli) de `depth` pasos
        desde `node`.  Retorna el reward del retorno directo si ningún candidato
        es factible o depth==0.
        """
        rm = _get_rm(t_elapsed)
        return_reward = rm[node, start_node]

        if depth == 0:
            return return_reward

        arrival_day = _day_index(start_day_idx, t_elapsed, max_day)
        lane_exists = draw_lane_availability(
            start_day_idx, node=node, arrival_day=arrival_day,
            avail_prob_arr=avail_prob_arr, num_nodes=num_n,
        )

        best_extended = -np.inf

        for cand in range(num_n):
            if cand == node or cand in visited_set:
                continue
            # Filtro Bernoulli — se omite cuando salimos desde start_node,
            # consistente con solve_heuristic_rolling_horizon_stochastic.
            if node != start_node and lane_exists[cand] != 1:
                continue
            step_time   = time_m[node, cand]
            return_time = time_m[cand, start_node]
            if t_elapsed + step_time + return_time > max_d:
                continue

            arc_reward    = rm[node, cand]
            t_after       = t_elapsed + step_time
            candidate_score = arc_reward + _lookahead_score(
                cand, t_after, visited_set | {cand}, depth - 1
            )

            if candidate_score > best_extended:
                best_extended = candidate_score

        return best_extended if best_extended > -np.inf else return_reward

    # ── Loop principal ────────────────────────────────────────────────────────
    current_node = start_node
    time_elapsed = 0.0
    route        = [start_node]
    visited      = {start_node}
    steps        = 0

    while steps < max_arcs:
        rm = _get_rm(time_elapsed)
        arrival_day = _day_index(start_day_idx, time_elapsed, max_day)
        lane_exists = draw_lane_availability(
            start_day_idx, node=current_node, arrival_day=arrival_day,
            avail_prob_arr=avail_prob_arr, num_nodes=num_n,
        )

        best_score     = -np.inf
        best_next_node = None

        for next_node in range(num_n):
            if next_node == current_node or next_node in visited:
                continue
            if current_node != start_node and lane_exists[next_node] != 1:
                continue
            step_time   = time_m[current_node, next_node]
            return_time = time_m[next_node, start_node]
            if time_elapsed + step_time + return_time > max_d:
                continue

            arc_reward = rm[current_node, next_node]
            t_after    = time_elapsed + step_time
            score = arc_reward + _lookahead_score(
                next_node, t_after, visited | {next_node}, lookahead - 1
            )

            if score > best_score:
                best_score     = score
                best_next_node = next_node

        if best_next_node is not None:
            time_elapsed += time_m[current_node, best_next_node]
            current_node  = best_next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break

    # Cierre de ciclo + validación + evaluación canónica (helper compartido).
    return _close_cycle_and_finalize(
        route, current_node, time_elapsed, start_node, time_m, max_d,
        start_day_idx, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=avail_prob_arr,
    )


def _rh_greedy_stoch_from_state(
    current_node, time_elapsed, visited,
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    avail_prob_arr, seed=None,
):
    """Greedy estocástico desde un estado arbitrario (current_node, time_elapsed, visited).

    Función auxiliar privada de solve_mc_rollout_stochastic. Replica la lógica de
    solve_heuristic_rolling_horizon_stochastic pero arranca desde un estado intermedio
    arbitrario en lugar de siempre desde (start_node, t=0, visited={start_node}).

    Retorna el reward acumulado desde current_node hasta cerrar el ciclo en start_node,
    usando el mismo filtrado Bernoulli y la misma lógica de día dinámico que el resto
    del sistema.

    NOTA (comportamiento actual, preservado en el refactor): el parámetro `seed`
    NO tiene efecto. La única fuente de aleatoriedad, draw_lane_availability, se
    siembra internamente a partir de (start_day_idx, node, arrival_day) y NO del
    RNG global ni de este `seed`. En consecuencia, las n_simulations trayectorias
    que solve_mc_rollout_stochastic lanza por candidato son IDÉNTICAS entre sí, y
    n_simulations no reduce varianza (solo multiplica el costo). Esto se documenta
    aquí a propósito; "arreglarlo" cambiaría las salidas y debe validarse aparte.
    """
    from problem_data import draw_lane_availability

    num_days = rate_stack.shape[0]
    max_day = num_days - 1
    max_arcs = num_n - 1

    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, max_day, as_numpy=True)

    node = current_node
    t = float(time_elapsed)
    vis = set(visited)
    steps = len(vis) - 1  # nodes visited so far (excluding start_node)
    total_reward = 0.0

    while steps < max_arcs:
        rm = _get_rm(t)
        arrival_day = _day_index(start_day_idx, t, max_day)
        lane_exists = draw_lane_availability(
            start_day_idx, node=node, arrival_day=arrival_day,
            avail_prob_arr=avail_prob_arr, num_nodes=num_n,
        )

        best_score = -np.inf
        best_next = None

        for nxt in range(num_n):
            if nxt == node or nxt in vis:
                continue
            if node != start_node and lane_exists[nxt] != 1:
                continue
            step_t = float(time_m[node, nxt])
            ret_t = float(time_m[nxt, start_node])
            if t + step_t + ret_t <= max_d:
                score = rm[node, nxt] + rm[nxt, start_node]
                if score > best_score:
                    best_score = score
                    best_next = nxt

        if best_next is not None:
            total_reward += rm[node, best_next]
            t += float(time_m[node, best_next])
            node = best_next
            vis.add(node)
            steps += 1
        else:
            break

    # Close cycle
    if node != start_node:
        ret_t = float(time_m[node, start_node])
        if t + ret_t <= max_d:
            total_reward += _get_rm(t)[node, start_node]

    return total_reward


def solve_mc_rollout_stochastic(
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    avail_prob_arr,
    n_simulations=30,
    lookahead_base_policy='rh_greedy',
):
    """Monte Carlo Rollout estocástico (VFA/policy rollout).

    En cada paso de decisión estima el Q-value de cada acción candidata j
    simulando n_simulations trayectorias completas desde el estado post-acción
    (j, t + time[i,j], visited ∪ {j}) usando la política base greedy estocástica
    (_rh_greedy_stoch_from_state). Selecciona la acción con mayor Q-value esperado.

    **Garantía teórica (policy improvement)**
    Por el teorema de mejora de política de Powell (Bertsekas & Castanon, 1999;
    Secomandi, 2001), el rollout domina en esperanza a la política base:
        V^rollout(s) ≥ V^base(s)  para todo estado s.
    La garantía es en esperanza; no está garantizada sample-by-sample con n finito.

    **Semillas de simulación**
    Cada trayectoria k desde candidato j usa seed = (start_day_idx*9973 + j*97 + k)
    & 0xFFFFFFFF para reproducibilidad. Los sorteos Bernoulli individuales de
    draw_lane_availability son siempre deterministas dado (start_day_idx, node, arrival_day),
    por lo que el seed controla solo el estado inicial del RNG de la política base.

    Parameters
    ----------
    n_simulations : int — trayectorias por candidato (default 30). Reduce varianza
                    de la estimación Q a costa de tiempo O(n_simulations * N).
    lookahead_base_policy : str — política base usada ('rh_greedy').

    Returns
    -------
    (status, route, total_reward, total_duration, is_valid)
    Mismo formato que todos los solvers estocásticos del sistema.
    reward recomputado con simulate_route_reward (evaluación canónica).
    """
    from problem_data import draw_lane_availability

    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    num_days = rate_stack.shape[0]
    max_day = num_days - 1
    max_arcs = num_n - 1

    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, max_day, as_numpy=True)

    current_node = start_node
    time_elapsed = 0.0
    route = [start_node]
    visited = {start_node}
    steps = 0

    while steps < max_arcs:
        rm = _get_rm(time_elapsed)
        arrival_day = _day_index(start_day_idx, time_elapsed, max_day)
        lane_exists = draw_lane_availability(
            start_day_idx, node=current_node, arrival_day=arrival_day,
            avail_prob_arr=avail_prob_arr, num_nodes=num_n,
        )

        # Candidatos factibles en tiempo con lane disponible
        candidates = []
        for j in range(num_n):
            if j == current_node or j in visited:
                continue
            if current_node != start_node and lane_exists[j] != 1:
                continue
            if (time_elapsed + float(time_m[current_node, j])
                    + float(time_m[j, start_node]) <= max_d):
                candidates.append(j)

        if not candidates:
            break

        # Estimar Q-value para cada candidato via n_simulations trayectorias
        best_q = -np.inf
        best_j = None

        for j in candidates:
            imm_reward = rm[current_node, j]
            t_after = time_elapsed + float(time_m[current_node, j])
            visited_after = visited | {j}

            future_rewards = []
            for k in range(n_simulations):
                seed = int((start_day_idx * _BERNOULLI_SEED_A
                            + j * _BERNOULLI_SEED_B + k) & 0xFFFFFFFF)
                future_r = _rh_greedy_stoch_from_state(
                    j, t_after, visited_after,
                    start_node, time_m, rate_stack, loads_stack,
                    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
                    avail_prob_arr, seed=seed,
                )
                future_rewards.append(future_r)

            q_value = imm_reward + float(np.mean(future_rewards))

            if q_value > best_q:
                best_q = q_value
                best_j = j

        time_elapsed += float(time_m[current_node, best_j])
        current_node = best_j
        route.append(current_node)
        visited.add(current_node)
        steps += 1

    # Cierre de ciclo + validación + evaluación canónica final (helper compartido).
    # Importante: se re-evalúa con simulate_route_reward, NO con los Q-values
    # acumulados durante el rollout.
    return _close_cycle_and_finalize(
        route, current_node, time_elapsed, start_node, time_m, max_d,
        start_day_idx, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=avail_prob_arr,
    )


def solve_mip_dynamic(
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    time_limit_s=None,
):
    """MIP exacto con reward dependiente del día de salida del arco.

    Formulación step-indexed (k = posición en la ruta):
      x[i,j,k]  — binaria: arco (i,j) en el paso k.
      z[k,b]     — binaria: el paso k parte en el bucket relativo b.
      y[i,j,k,b] — continua [0,1]: linealización de x[i,j,k]·z[k,b].

    El reward del arco (i,j) en el paso k usa la matriz del día
    d(b) = min(start_day_idx + b, max_day), donde b es el bucket
    elegido por z[k,b].  El día relevante es el de **salida** del arco,
    replicando exactamente _day_index(start_day_idx, T_k, max_day).

    NOTA: este MIP es cota superior **solo del track determinista**
    (avail_prob_arr=None).  No es cota para el mundo estocástico.

    Si `time_limit_s` corta antes de probar optimalidad, el status es
    "TimeLimit" y **no** hay garantía de cota superior para esa instancia.
    """
    import pulp  # import perezoso: única función del módulo que usa pulp (además de solve_lp_relaxation)
    from problem_data import build_day_matrices
    from config import MIP_FLOOR_EPS
    # DAYS_PER_PERIOD es constante del módulo Solvers (línea ~5)

    # ── Preprocesamiento ──────────────────────────────────────────────────────
    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    max_day = rate_stack.shape[0] - 1

    # Buckets relativos: b ∈ {0, …, B-1}
    B = int(max_d // DAYS_PER_PERIOD) + 1   # con max_d=77 → B=6

    # Caché de matrices de reward numpy (penalizadas) por índice efectivo de día.
    _rm_cache = {}
    def _get_R(b):
        d_eff = min(start_day_idx + b, max_day)
        if d_eff not in _rm_cache:
            _, rm_pen = build_day_matrices(
                rate_stack[d_eff], loads_stack[d_eff], distance_arr, diesel_arr
            )
            _rm_cache[d_eff] = np.array(rm_pen, dtype=float)
        return _rm_cache[d_eff]

    K = num_n  # pasos máximos (= num_n arcos: num_n-1 intermedios + retorno)

    # Arcos permitidos: sin self-loops, factibles en tiempo, y que permitan retorno.
    arcs = []
    for i in range(num_n):
        for j in range(num_n):
            if i == j:
                continue
            if time_m[i, j] > max_d:
                continue
            # j intermedio: debe permitir volver al depósito
            if j != start_node and time_m[i, j] + time_m[j, start_node] > max_d:
                continue
            arcs.append((i, j))

    # ── Modelo ────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem(f"MIP_dynamic_s{start_node}_d{start_day_idx}", pulp.LpMaximize)

    # Variables x[i,j,k]
    x = {}
    for (i, j) in arcs:
        for k in range(K):
            # En k=0 solo puede salir start_node; en k≥1 start_node no puede salir.
            if k == 0 and i != start_node:
                continue
            if k >= 1 and i == start_node:
                continue
            # En k=K-1 el destino debe ser start_node.
            if k == K - 1 and j != start_node:
                continue
            x[i, j, k] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    # Variables z[k,b]
    z = {}
    for k in range(K):
        for b in range(B):
            z[k, b] = pulp.LpVariable(f"z_{k}_{b}", cat='Binary')

    # Variables y[i,j,k,b] continuas en [0,1]
    y = {}
    for (k_var, xvar) in x.items():
        i, j, k = k_var
        for b in range(B):
            y[i, j, k, b] = pulp.LpVariable(f"y_{i}_{j}_{k}_{b}", 0, 1, cat='Continuous')

    # ── Objetivo ──────────────────────────────────────────────────────────────
    prob += pulp.lpSum(
        _get_R(b)[i, j] * y[i, j, k, b]
        for (i, j, k, b) in y
    )

    # ── Restricciones ─────────────────────────────────────────────────────────

    # 1. Anclaje al depósito: exactamente un arco desde start_node en k=0.
    prob += (
        pulp.lpSum(x[start_node, j, 0] for (si, j, k) in x if si == start_node and k == 0) == 1,
        "depot_depart_k0"
    )

    # 2. Un arco por paso, sin huecos.
    for k in range(K):
        arcs_k = [(i, j) for (i, j, kk) in x if kk == k]
        prob += (
            pulp.lpSum(x[i, j, k] for (i, j) in arcs_k) <= 1,
            f"one_arc_step_{k}"
        )
    for k in range(K - 1):
        arcs_k  = [(i, j) for (i, j, kk) in x if kk == k]
        arcs_k1 = [(i, j) for (i, j, kk) in x if kk == k + 1]
        prob += (
            pulp.lpSum(x[i, j, k + 1] for (i, j) in arcs_k1)
            <= pulp.lpSum(x[i, j, k] for (i, j) in arcs_k),
            f"no_gap_{k}"
        )

    # 3. Continuidad de flujo para nodos intermedios.
    for j in range(num_n):
        if j == start_node:
            continue
        for k in range(K - 1):
            in_j_k  = [(i, j) for (i, jj, kk) in x if jj == j and kk == k]
            out_j_k1 = [(j, l) for (jj, l, kk) in x if jj == j and kk == k + 1]
            if in_j_k or out_j_k1:
                prob += (
                    pulp.lpSum(x[j, l, k + 1] for (_, l) in out_j_k1)
                    == pulp.lpSum(x[i, j, k] for (i, _) in in_j_k),
                    f"flow_{j}_{k}"
                )

    # 4. Retorno único al depósito.
    in_depot = [(i, start_node, k) for (i, j, k) in x if j == start_node]
    prob += (
        pulp.lpSum(x[i, start_node, k] for (i, _, k) in in_depot) == 1,
        "return_once"
    )

    # 5. Sin revisitas de nodos intermedios.
    for j in range(num_n):
        if j == start_node:
            continue
        arcs_to_j = [(i, j, k) for (i, jj, k) in x if jj == j]
        if arcs_to_j:
            prob += (
                pulp.lpSum(x[i, j, k] for (i, _, k) in arcs_to_j) <= 1,
                f"no_revisit_{j}"
            )

    # 6. Tiempo total.
    prob += (
        pulp.lpSum(time_m[i, j] * x[i, j, k] for (i, j, k) in x) <= max_d,
        "time_limit"
    )

    # 7. Asignación de bucket con big-M = max_d.
    M_bm = max_d
    for k in range(K):
        arcs_k = [(i, j) for (i, j, kk) in x if kk == k]
        step_used = pulp.lpSum(x[i, j, k] for (i, j) in arcs_k)

        # Σ_b z[k,b] = step_used (un bucket exactamente si el paso se usa)
        prob += (
            pulp.lpSum(z[k, b] for b in range(B)) == step_used,
            f"bucket_assign_{k}"
        )

        # Tiempo acumulado al inicio del paso k: T_k = Σ_{k'<k} Σ_{i,j} time[i,j]*x[i,j,k']
        T_k_expr = pulp.lpSum(
            time_m[i, j] * x[i, j, kp]
            for (i, j, kp) in x if kp < k
        ) if k > 0 else 0

        for b in range(B):
            # T_k >= b * DAYS_PER_PERIOD - M*(1 - z[k,b])
            prob += (
                T_k_expr >= b * DAYS_PER_PERIOD - M_bm * (1 - z[k, b]),
                f"bucket_lb_{k}_{b}"
            )
            # T_k <= (b+1)*DAYS_PER_PERIOD - ε + M*(1 - z[k,b])
            prob += (
                T_k_expr <= (b + 1) * DAYS_PER_PERIOD - MIP_FLOOR_EPS + M_bm * (1 - z[k, b]),
                f"bucket_ub_{k}_{b}"
            )

    # 8. Linealización completa de y.
    for (i, j, k, b) in y:
        prob += (y[i, j, k, b] <= x[i, j, k],               f"y_le_x_{i}_{j}_{k}_{b}")
        prob += (y[i, j, k, b] <= z[k, b],                   f"y_le_z_{i}_{j}_{k}_{b}")
        prob += (y[i, j, k, b] >= x[i, j, k] + z[k, b] - 1, f"y_ge_xz_{i}_{j}_{k}_{b}")

    # 9. Desigualdad válida: bucket no decreciente en pasos consecutivos.
    for k in range(K - 1):
        arcs_k  = [(i, j) for (i, j, kk) in x if kk == k]
        arcs_k1 = [(i, j) for (i, j, kk) in x if kk == k + 1]
        step_used_k1 = pulp.lpSum(x[i, j, k + 1] for (i, j) in arcs_k1)
        prob += (
            pulp.lpSum(b * z[k + 1, b] for b in range(B))
            >= pulp.lpSum(b * z[k, b] for b in range(B)) - M_bm * (1 - step_used_k1),
            f"bucket_nondec_{k}"
        )

    # ── Resolve ───────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    prob.solve(solver)

    lp_status = pulp.LpStatus[prob.status]

    # ── Post-procesamiento: reconstrucción de ruta ────────────────────────────
    def _extract_route():
        route = [start_node]
        current = start_node
        for k in range(K):
            moved = False
            for (i, j) in [(i, j) for (i, j, kk) in x if kk == k and i == current]:
                if pulp.value(x[i, j, k]) is not None and pulp.value(x[i, j, k]) > 0.5:
                    route.append(j)
                    current = j
                    moved = True
                    break
            if not moved or current == start_node:
                break
        return route

    has_incumbente = prob.sol_status is not None and prob.sol_status >= 1

    if not has_incumbente:
        return MetaResult("Infeasible", None, -np.inf, np.inf)

    route = _extract_route()

    # Validación estructural
    route_valid = (
        len(route) >= 2
        and route[0] == start_node
        and route[-1] == start_node
        and len(set(route[1:-1])) == len(route[1:-1])
        and sum(time_m[route[p], route[p + 1]] for p in range(len(route) - 1)) <= max_d + 1e-6
    )

    if not route_valid:
        status_out = "TimeLimit" if lp_status != 'Optimal' else "Infeasible"
        return MetaResult(status_out, None, -np.inf, np.inf)

    # Re-evaluación canónica obligatoria
    canon_reward, canon_duration = simulate_route_reward(
        route, start_node, start_day_idx,
        time_m, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )

    # Determinar status de salida
    if lp_status == 'Optimal':
        status_out = "Optimal"
    else:
        status_out = "TimeLimit"

    # Verificación de paridad MIP vs canónica
    mip_obj = pulp.value(prob.objective)
    if mip_obj is not None and abs(mip_obj - canon_reward) > 1e-4:
        import warnings as _w
        _w.warn(
            f"MIP obj ({mip_obj:.4f}) != canonical reward ({canon_reward:.4f}) "
            f"| start_node={start_node} start_day_idx={start_day_idx} "
            f"diff={abs(mip_obj - canon_reward):.2e}"
        )

    # Verificación de paridad de buckets
    t_acc = 0.0
    for step, (a, b_node) in enumerate(zip(route[:-1], route[1:])):
        b_mip = _day_index(start_day_idx, t_acc, max_day) - start_day_idx
        b_mip = max(0, min(b_mip, B - 1))
        # bucket elegido por z
        b_chosen = None
        for b in range(B):
            zval = pulp.value(z[step, b])
            if zval is not None and zval > 0.5:
                b_chosen = b
                break
        if b_chosen is not None and b_chosen != b_mip:
            import warnings as _w
            _w.warn(
                f"BoundaryMismatch: start_node={start_node} start_day_idx={start_day_idx} "
                f"step={step} T_k={t_acc:.6f} bucket_canonical={b_mip} bucket_mip={b_chosen}"
            )
            status_out = "BoundaryMismatch"
        t_acc += time_m[a, b_node]

    return MetaResult(status_out, route, canon_reward, canon_duration)


def solve_heuristic_rolling_horizon_stochastic(
    start_node, time_m, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx, avail_prob_arr,
):
    """Variante estocástica de solve_heuristic_rolling_horizon.

    Idéntica a la versión determinista salvo que en cada paso descarta los
    nodos cuya lane está bloqueada según draw_lane_availability (mundo Bernoulli).
    La evaluación final usa simulate_route_reward con avail_prob_arr.
    """
    from problem_data import draw_lane_availability

    num_days = rate_stack.shape[0]
    # DataFrame (as_numpy=False): indexa rm[current_node][next_node] como la
    # versión greedy determinista, así que conserva ese tipo/acceso.
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, num_days - 1, as_numpy=False)

    if hasattr(time_m, 'iloc'):
        time_m = np.array(time_m, dtype=float)

    current_node = start_node
    time_elapsed = 0.0
    route = [start_node]
    visited = {start_node}
    steps = 0
    max_arcs = num_n - 1

    while steps < max_arcs:
        rm = _get_rm(time_elapsed)
        current_arrival_day = _day_index(start_day_idx, time_elapsed, num_days - 1)
        lane_exists = draw_lane_availability(
            start_day_idx, node=current_node, arrival_day=current_arrival_day,
            avail_prob_arr=avail_prob_arr, num_nodes=num_n,
        )

        best_score = -np.inf
        best_next_node = None

        for next_node in range(num_n):
            if next_node != current_node and next_node not in visited:
                # No aplicar filtro Bernoulli desde el nodo de inicio, igual que
                # beam_search_dynamic, que omite lane filtering cuando
                # current_node == start_node.
                if current_node != start_node and lane_exists[next_node] != 1:
                    continue
                step_time = time_m[current_node][next_node]
                return_time = time_m[next_node][start_node]
                if time_elapsed + step_time + return_time <= max_d:
                    score = rm[current_node][next_node] + rm[next_node][start_node]
                    if score > best_score:
                        best_score = score
                        best_next_node = next_node

        if best_next_node is not None:
            time_elapsed += time_m[current_node][best_next_node]
            current_node = best_next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            break

    # Cierre de ciclo + validación + evaluación canónica (helper compartido).
    return _close_cycle_and_finalize(
        route, current_node, time_elapsed, start_node, time_m, max_d,
        start_day_idx, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=avail_prob_arr,
    )
