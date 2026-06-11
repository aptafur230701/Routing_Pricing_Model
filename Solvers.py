import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical # For sampling actions
import matplotlib.pyplot as plt
import pulp


def generate_optimal_route_pytorch(agent, start_node, time_matrix, reward_matrix, NUM_NODES,MAX_DURATION,MAX_STEPS_PER_EPISODE):
        """
        Generates a route using the learned policy (greedy selection),
        preventing revisits to intermediate nodes.
        If max_steps is reached, attempts forced return if valid.
        Validates final duration window.
        """
        max_steps=MAX_STEPS_PER_EPISODE
        agent.epsilon = 0
        agent.policy_net.eval()
        current_node = start_node
        time_elapsed = 0.0
        state = np.array([current_node, time_elapsed / MAX_DURATION], dtype=np.float32)
        route = [start_node]
        visited_intermediate_nodes = set() # Keep track of nodes visited *other than* start_node
        total_reward = 0.0
        returned_home = False

        with torch.no_grad():
            for step in range(max_steps):
                # --- Action Selection ---
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                q_values_numpy = q_values.cpu().data.numpy()[0]

                # --- Masking Invalid Actions ---
                # 1. Don't stay in the same node
                q_values_numpy[current_node] = -np.inf

                # 2. Don't visit intermediate nodes already visited
                for visited_node_idx in visited_intermediate_nodes:
                    if 0 <= visited_node_idx < len(q_values_numpy): # Bounds check
                        q_values_numpy[visited_node_idx] = -np.inf

                # --- Choose Best Valid Action ---
                next_node = np.argmax(q_values_numpy)

                # Check if any valid action exists
                if q_values_numpy[next_node] == -np.inf:
                    # No valid moves possible (maybe all unvisited nodes violate time or Q-values are terrible)
                    # Try forcing return home immediately if possible
                    # print(f"DRL: Stuck at node {current_node}. No valid non-visited moves. Trying return home.")
                    if current_node != start_node:
                        return_time = time_matrix[current_node][start_node]
                        if time_elapsed + return_time <= MAX_DURATION + 1e-6:
                            next_node = start_node # Override choice to return home
                            # print("DRL: Forcing return home as only option.")
                        else:
                            # print(f"DRL Error: Stuck at node {current_node}. Cannot return home within duration.")
                            returned_home = False
                            break # Cannot proceed
                    else: # Stuck at start node? Should not happen if mask works.
                        returned_home = False
                        break


                # --- Simulate Step ---
                step_time = time_matrix[current_node][next_node]
                step_reward = reward_matrix[current_node][next_node]

                # --- Check immediate time violation (should be less likely now with stuck check) ---
                if time_elapsed + step_time > MAX_DURATION + 1e-6 and next_node != start_node:
                    # print(f"DRL: Next step to {next_node} violates MAX_DURATION. Stopping.")
                    returned_home = False
                    break

                # --- Update State ---
                time_elapsed += step_time
                total_reward += step_reward
                current_node = next_node
                route.append(current_node)
                # Add to visited set *only if* it's not the start node
                if current_node != start_node:
                    visited_intermediate_nodes.add(current_node)

                state = np.array([current_node, min(time_elapsed, MAX_DURATION) / MAX_DURATION], dtype=np.float32)

                # --- Check for Natural Return ---
                if current_node == start_node:
                    returned_home = True
                    break

            # --- End of Step Loop ---

            # --- Handle Forced Return if max_steps reached ---
            # (This logic might be less necessary now but keep as fallback)
            if not returned_home and current_node != start_node:
                # print(f"DRL: Max steps reached, attempting forced return from {current_node} to {start_node}")
                return_time = time_matrix[current_node][start_node]
                return_reward = reward_matrix[current_node][start_node]
                if time_elapsed + return_time <= MAX_DURATION + 1e-6:
                    time_elapsed += return_time
                    total_reward += return_reward
                    current_node = start_node
                    route.append(start_node)
                    returned_home = True
                # else: # Forced return violates time
                    # returned_home remains False

        # --- Final Validation ---
        agent.policy_net.train()

        is_cycle = returned_home and route[0] == start_node and route[-1] == start_node and len(route)>1

        if not is_cycle:
            return None, -np.inf, np.inf # Failed route

        is_valid_duration = time_elapsed <= MAX_DURATION

        # Check for duplicate intermediate nodes (should be prevented by logic above)
        intermediate_nodes = route[1:-1]
        has_duplicates = len(intermediate_nodes) != len(set(intermediate_nodes))
        if has_duplicates:
            print(f"Warning: DRL route {route} has duplicate intermediate nodes despite masking!")
            # Treat as invalid? Or just note it. Let's return it but validity check below will fail if needed.

        if is_valid_duration and not has_duplicates:
            return route, total_reward, time_elapsed # Valid cycle found
        else:
            # Cycle formed, but duration or node visit is invalid
            return route, total_reward, time_elapsed # Return invalid route details

def solve_mip(start_node, time_m, reward_m, max_d, num_n):
    """Solves the VRP variant using MIP for a given start node."""
    nodes = list(range(num_n))
    other_nodes = [n for n in nodes if n != start_node]

    # Create the model
    prob = pulp.LpProblem(f"VRP_Cycle_{start_node}", pulp.LpMaximize)

    # Decision Variables
    x = pulp.LpVariable.dicts("Route", (nodes, nodes), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("MTZ", nodes, 1, num_n - 1, pulp.LpContinuous)

    # Objective Function
    prob += pulp.lpSum(reward_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    # Constraints
    # 1. Degree Constraints
    for k in nodes:
        prob += pulp.lpSum(x[k][j] for j in nodes if k != j) == pulp.lpSum(x[j][k] for j in nodes if k != j)
        if k == start_node:
            prob += pulp.lpSum(x[start_node][j] for j in nodes if j != start_node) == 1
            prob += pulp.lpSum(x[j][start_node] for j in nodes if j != start_node) == 1
        else:
             prob += pulp.lpSum(x[j][k] for j in nodes if j != k) <= 1

    # 2. Duration Constraints
    total_time = pulp.lpSum(time_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)
    prob += total_time <= max_d

    # 3. Subtour Elimination (MTZ)
    for i in other_nodes:
        prob += u[i] >= 1
        for j in other_nodes:
            if i != j:
                 prob += u[i] - u[j] + 1 <= (num_n - 1) * (1 - x[i][j])

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    # Extract results
    status = pulp.LpStatus[prob.status]
    route = None
    total_reward = -np.inf
    total_duration = np.inf

    if status == 'Optimal':
        total_reward = pulp.value(prob.objective)
        total_duration = pulp.value(total_time)

        # --- Modified Route Reconstruction ---
        try: # Add a try-except block for safety during reconstruction
            current_node = start_node
            route = [start_node]
            visited_count = 0 # Safety counter

            while visited_count <= num_n: # Limit search depth
                found_next = False
                for j in nodes:
                    # Check if arc variable exists, is not None, and is selected (> 0.99)
                    # Also ensure j is not the current node
                    if j != current_node and \
                       x[current_node][j] is not None and \
                       x[current_node][j].varValue is not None and \
                       x[current_node][j].varValue > 0.99:

                        route.append(j)
                        current_node = j
                        found_next = True
                        break # Move to the next node in the path

                visited_count += 1

                if current_node == start_node: # Successfully completed the cycle
                    break
                if not found_next: # Dead end found during reconstruction
                    # print(f"MIP Route Reconstruction Error: Dead end at node {current_node} for start {start_node}.")
                    route = None # Invalid route
                    break
                if visited_count > num_n: # Avoid infinite loops / too many steps
                    # print(f"MIP Route Reconstruction Error: Route too long for start {start_node}. Path: {route}")
                    route = None # Invalid route
                    break

            # Final validation of reconstructed route
            if route is None or route[0] != start_node or route[-1] != start_node:
                 # print(f"MIP Route Reconstruction resulted in invalid path for start {start_node}. Route: {route}")
                 route = None
                 # If route is invalid, reset reward/duration derived from MIP objective
                 total_reward = -np.inf
                 total_duration = np.inf
                 status = 'Error_In_Route' # Update status to reflect this

        except Exception as e:
            print(f"Exception during MIP route reconstruction for start {start_node}: {e}")
            route = None
            total_reward = -np.inf
            total_duration = np.inf
            status = 'Error_Exception'
        # --- End of Modified Route Reconstruction ---

    # Ensure reward/duration are consistent if route is None
    if route is None:
         total_reward = -np.inf
         total_duration = np.inf
         # Update status if it was 'Optimal' but route failed
         if status == 'Optimal': status = 'Optimal_Route_Fail'

    return status, route, total_reward, total_duration

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
    return status, route, total_reward, time_elapsed, is_valid


def solve_lp_relaxation(start_node, time_m, reward_m, max_d, num_n):
    """Solves LP relaxation of the VRP variant for upper bound."""
    nodes = list(range(num_n))
    other_nodes = [n for n in nodes if n != start_node]

    # Create LP model
    lp_prob = pulp.LpProblem(f"VRP_LP_Relaxation_{start_node}", pulp.LpMaximize)

    # Decision Variables (continuous between 0 and 1)
    x = pulp.LpVariable.dicts("Route", (nodes, nodes), 0, 1, pulp.LpContinuous)
    u = pulp.LpVariable.dicts("MTZ", nodes, 1, num_n - 1, pulp.LpContinuous)

    # Same objective and constraints as MIP
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
        return status, route, total_reward, total_time, is_valid

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
    return status, best_route, best_reward, best_time, is_valid


def solve_LNS_metaheuristic(start_node, time_m, reward_m, max_d, num_n, seed=None):
    """
    Large Neighborhood Search (LNS) metaheuristic:
    1. Start with a greedy initial solution
    2. Iteratively destroy and repair neighborhoods
    3. Accept solutions if they improve best known or pass probabilistic criterion
    Returns: status, route, total_reward, total_duration (same format as solve_mip)
    """
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
            return "Infeasible", None, -np.inf, np.inf
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

    return final_status, best_route, best_reward, best_time


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
                score = delta_r - 0.001 * max(delta_t, 0)

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
        return "Infeasible", None, -np.inf, np.inf

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

    return final_status, best_overall['route'], best_overall['reward'], best_overall['duration']


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
    nodes1 = set(route1[1:-1])
    nodes2 = set(route2[1:-1])

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
    population is evaluated on the same footing as MIP-Exact and DRL Det.
    Search operators (crossover, LNS repair, 2-opt) still use the static day-0
    reward matrix for fast local decisions.

    Returns: status, route, total_reward, total_duration  (same format as solve_mip)
    """
    from problem_data import build_day_matrices
    _, reward_m = build_day_matrices(
        rate_stack[start_day_idx], loads_stack[start_day_idx], distance_arr, diesel_arr
    )
    time_m_np = np.array(time_m, dtype=float)

    # Pre-cache reward matrices as numpy arrays so _eval avoids pandas overhead.
    # Semantically identical to simulate_route_reward(..., avail_prob_arr=None).
    max_day = rate_stack.shape[0] - 1
    _max_offset = int(max_d // 14) + 2
    _rm_cache: dict = {}
    for _d in range(start_day_idx, min(start_day_idx + _max_offset, max_day + 1)):
        _, _rm_df = build_day_matrices(rate_stack[_d], loads_stack[_d], distance_arr, diesel_arr)
        _rm_cache[_d] = _rm_df.to_numpy()
    if max_day not in _rm_cache:
        _, _rm_df = build_day_matrices(rate_stack[max_day], loads_stack[max_day], distance_arr, diesel_arr)
        _rm_cache[max_day] = _rm_df.to_numpy()

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
        """Fast canonical fitness: pre-cached numpy matrices, sequential arrival days."""
        if route is None or len(route) < 2:
            return -np.inf, np.inf
        t = 0.0
        r = 0.0
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            d = min(start_day_idx + int(t // 14), max_day)
            r += _rm_cache[d][i, j]
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

    if not population:
        # Emergency fallback: greedy/NN/random all rejected every arc (e.g. all
        # rewards negative on day_0). Sweep every direct out-and-back using
        # time-only feasibility — mirrors solve_LNS_metaheuristic fallback logic.
        for _nb in range(num_n):
            if _nb == start_node:
                continue
            _t = float(time_m_np[start_node, _nb] + time_m_np[_nb, start_node])
            if _t <= max_d:
                _fb = [start_node, _nb, start_node]
                _r, _d = _eval(_fb)
                population.append({'route': _fb, 'reward': _r, 'duration': _d})
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
                if child_dur <= max_d:
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
        return final_status, None, -np.inf, np.inf

    # Canonical re-evaluation via simulate_route_reward guarantees bit-exact
    # parity with MIP-Exact and all other solvers. The internal _eval uses
    # pre-cached numpy arrays (_rm_cache) whose float accumulation can diverge
    # slightly from simulate_route_reward, causing the winner to appear above
    # the MIP-Exact bound. Passing avail_prob_arr=None keeps this deterministic.
    total_reward, total_duration = simulate_route_reward(
        best_overall['route'], start_node, start_day_idx,
        time_m_np, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    return final_status, best_overall['route'], total_reward, total_duration


def _oracle_matrix_from_days(
    arrival_days, start_node, rate_stack, loads_stack,
    distance_arr, diesel_arr, start_day_idx, avail_prob_arr,
    num_nodes, max_day,
):
    """Build the oracle reward matrix given a pre-computed per-node arrival-day array.

    Arcs from/to start_node are never Bernoulli-filtered, matching
    beam_search_dynamic which skips lane filtering when current_node == start_node.
    Blocked arcs (lane_exists[j] == 0) get BIG_M_PENALTY so the MIP avoids them.
    """
    from config import MPG, MARGINAL_COST_SIN_DIESEL, BIG_M_PENALTY

    unique_days = np.unique(arrival_days)
    reward_by_day = {}
    for d in unique_days:
        revenue = (rate_stack[d] * distance_arr).copy()
        revenue[loads_stack[d] <= 1] = 0
        cost = distance_arr * (diesel_arr / MPG) + distance_arr * MARGINAL_COST_SIN_DIESEL
        reward_by_day[d] = np.round(revenue - cost, 0)

    oracle_r = np.full((num_nodes, num_nodes), BIG_M_PENALTY, dtype=np.float64)
    for i in range(num_nodes):
        arrival_day_i = int(arrival_days[i])
        seed = int((start_day_idx * 9973 + i * 97 + arrival_day_i) & 0xFFFFFFFF)
        rng = np.random.default_rng(seed)
        lane_exists = (rng.random(num_nodes) < avail_prob_arr[i]).astype(np.int8)
        for j in range(num_nodes):
            if i == j:
                oracle_r[i, j] = BIG_M_PENALTY
            elif i == start_node or j == start_node:
                oracle_r[i, j] = reward_by_day[arrival_day_i][i, j]
            elif lane_exists[j] == 1:
                oracle_r[i, j] = reward_by_day[arrival_day_i][i, j]
            # else: lane_exists[j] == 0 → stays BIG_M_PENALTY

    return oracle_r


def build_oracle_reward_matrix(
    start_node, time_matrix_np, rate_stack, loads_stack,
    distance_arr, diesel_arr, start_day_idx, avail_prob_arr,
    num_nodes, max_day,
):
    """Oracle reward matrix using Dijkstra arrival days (public helper, kept for compat).

    solve_mip_oracle uses _oracle_matrix_from_days directly with iteratively
    refined arrival days; this function is retained as a convenience entry point.
    """
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    sparse_tm    = csr_matrix(time_matrix_np)
    dist_matrix  = shortest_path(sparse_tm, method='D', directed=True, indices=start_node)
    arrival_days = np.array([
        min(start_day_idx + int(dist_matrix[i] // 14), max_day)
        for i in range(num_nodes)
    ], dtype=np.int32)

    return _oracle_matrix_from_days(
        arrival_days, start_node, rate_stack, loads_stack,
        distance_arr, diesel_arr, start_day_idx, avail_prob_arr,
        num_nodes, max_day,
    )


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
    from problem_data import build_day_matrices, draw_lane_availability

    if route is None or len(route) < 2:
        return -np.inf, np.inf

    max_day   = rate_stack.shape[0] - 1
    num_nodes = distance_arr.shape[0]
    day_cache = {}

    def _get_rm(day_idx):
        if day_idx not in day_cache:
            _, rm = build_day_matrices(
                rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
            )
            day_cache[day_idx] = rm
        return day_cache[day_idx]

    time_elapsed = 0.0
    total_reward = 0.0

    for step in range(len(route) - 1):
        i       = route[step]
        j       = route[step + 1]
        day_idx = min(start_day_idx + int(time_elapsed // 14), max_day)
        rm      = _get_rm(day_idx)
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

    return total_reward, time_elapsed


def solve_mip_oracle(
    start_node, time_matrix_np, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx, avail_prob_arr,
    max_iter=5,
):
    """MIP oracle with iterative arrival-day refinement.

    Iteration 0 uses Dijkstra arrival days as the initial estimate.  After each
    MIP solve the chosen route is walked sequentially; the real arrival day for
    every visited node is fed back into arrival_days and the MIP is re-solved.
    The loop stops when the route stops changing (fixed point) or max_iter is
    reached.  At convergence, the Bernoulli seeds used inside the oracle matrix
    match exactly the seeds that draw_lane_availability would produce during a
    DRL Real rollout following the same route — eliminating the Dijkstra-day
    inconsistency.  Nodes never visited by the converged route retain Dijkstra
    days (they were evaluated but not selected, so their seeds don't affect the
    final result).

    The final reported reward is computed by simulate_route_reward with real
    sequential days and Bernoulli filtering, keeping the Oracle on the same
    stochastic footing as DRL Real.
    """
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    max_day     = rate_stack.shape[0] - 1
    nodes       = list(range(num_n))
    other_nodes = [n for n in nodes if n != start_node]

    # ── inner helpers ────────────────────────────────────────────────────────

    def _run_mip(oracle_r):
        """Build and solve the MIP for one iteration; return (status_str, route|None)."""
        prob = pulp.LpProblem(f"VRP_Oracle_{start_node}", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("Route", (nodes, nodes), 0, 1, pulp.LpBinary)
        u = pulp.LpVariable.dicts("MTZ",   nodes, 1, num_n - 1, pulp.LpContinuous)

        prob += pulp.lpSum(
            oracle_r[i][j] * x[i][j] for i in nodes for j in nodes if i != j
        )
        for k in nodes:
            prob += (
                pulp.lpSum(x[k][j] for j in nodes if k != j)
                == pulp.lpSum(x[j][k] for j in nodes if k != j)
            )
            if k == start_node:
                prob += pulp.lpSum(x[start_node][j] for j in nodes if j != start_node) == 1
                prob += pulp.lpSum(x[j][start_node] for j in nodes if j != start_node) == 1
            else:
                prob += pulp.lpSum(x[j][k] for j in nodes if j != k) <= 1

        total_time = pulp.lpSum(
            time_matrix_np[i][j] * x[i][j] for i in nodes for j in nodes if i != j
        )
        prob += total_time <= max_d

        for i in other_nodes:
            prob += u[i] >= 1
            for j in other_nodes:
                if i != j:
                    prob += u[i] - u[j] + 1 <= (num_n - 1) * (1 - x[i][j])

        pulp.PULP_CBC_CMD(msg=0).solve(prob)
        status = pulp.LpStatus[prob.status]
        if status != 'Optimal':
            return status, None

        try:
            cur = start_node
            route = [start_node]
            for _ in range(num_n + 1):
                moved = False
                for j in nodes:
                    v = x[cur][j].varValue if (x[cur][j] is not None
                                               and x[cur][j].varValue is not None) else 0.0
                    if j != cur and v > 0.99:
                        route.append(j)
                        cur = j
                        moved = True
                        break
                if cur == start_node:
                    break
                if not moved:
                    return 'Error_In_Route', None
            if route[0] != start_node or route[-1] != start_node:
                return 'Error_In_Route', None
            return status, route
        except Exception as exc:
            print(f"  [Oracle] route reconstruction error (start={start_node}): {exc}")
            return 'Error_Exception', None

    def _sequential_days(route):
        """Return {node: real_arrival_day} for every node in route."""
        days = {}
        t = 0.0
        for step in range(len(route) - 1):
            i = route[step]
            days[i] = min(start_day_idx + int(t // 14), max_day)
            t += float(time_matrix_np[i][route[step + 1]])
        days[route[-1]] = min(start_day_idx + int(t // 14), max_day)
        return days

    # ── initialise arrival_days with Dijkstra estimate ───────────────────────

    sparse_tm    = csr_matrix(time_matrix_np)
    dist_m       = shortest_path(sparse_tm, method='D', directed=True, indices=start_node)
    arrival_days = np.array([
        min(start_day_idx + int(dist_m[i] // 14), max_day)
        for i in range(num_n)
    ], dtype=np.int32)

    # ── iterative fixed-point loop ───────────────────────────────────────────

    prev_route = None
    status     = 'Infeasible'
    route      = None

    for _it in range(max_iter):
        oracle_r       = _oracle_matrix_from_days(
            arrival_days, start_node, rate_stack, loads_stack,
            distance_arr, diesel_arr, start_day_idx, avail_prob_arr,
            num_n, max_day,
        )
        status, route = _run_mip(oracle_r)

        if route is None:
            break

        if route == prev_route:
            # Fixed point reached: arrival_days already reflect this route.
            break

        # Refine arrival_days for every node on the route with real sequential
        # days.  Nodes not on the route keep their current (Dijkstra) estimate.
        for node, real_day in _sequential_days(route).items():
            arrival_days[node] = real_day

        prev_route = route

    # ── final reward evaluation ──────────────────────────────────────────────

    if route is None or len(route) < 2:
        return status if status != 'Optimal' else 'Infeasible', None, -np.inf, np.inf

    total_reward, total_duration = simulate_route_reward(
        route, start_node, start_day_idx,
        time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=avail_prob_arr,
    )
    return status, route, total_reward, total_duration


def solve_mip_exact(
    start_node:     int,
    time_matrix_np: np.ndarray,
    rate_stack:     np.ndarray,
    loads_stack:    np.ndarray,
    distance_arr:   np.ndarray,
    diesel_arr:     np.ndarray,
    max_d:          float,
    num_n:          int,
    start_day_idx:  int,
    avail_prob_arr: np.ndarray = None,   # accepted but intentionally ignored
) -> tuple:
    """Exact time-indexed MIP — deterministic optimal upper bound.

    Solves the routing problem on the *deterministic* world (no Bernoulli lane
    filtering), so the returned reward is a provable upper bound for any solver
    evaluated on the same deterministic reward matrices.

    Day-partition formulation
    -------------------------
    K_max = floor(max_d / 14) gives at most K_max+1 ≤ 6 distinct day offsets.
    Binary z[i,j,k] = 1 iff arc (i→j) is active AND int(t_i // 14) == k.
    Continuous t[i] tracks the *exact* cumulative arrival time at each visited
    node: tight lower AND upper bounds jointly enforce t[j] = t[i] + time[i,j]
    on every active arc, preventing the solver from inflating t[i] to claim a
    more profitable day slot.  The t propagation also eliminates subtours
    (any non-depot cycle implies sum-of-times ≥ 0, contradiction).

    Returns
    -------
    (status, route, total_reward, total_duration)
    reward recomputed via simulate_route_reward(..., avail_prob_arr=None)
    for bit-exact parity with the DRL deterministic evaluation.
    """
    from problem_data import build_day_matrices

    max_day     = rate_stack.shape[0] - 1
    nodes       = list(range(num_n))
    other_nodes = [n for n in nodes if n != start_node]

    K_max = int(max_d // 14)        # 5 for max_d ≈ 77
    k_set = list(range(K_max + 1))
    # BIG_M_T: big-M for time-propagation constraints (C4).
    # Must be ≥ max_d + max(time[i,j]) so that inactive arcs impose no
    # binding lower bound on arrival times of non-visited nodes.
    BIG_M_T = float(max_d) + float(time_matrix_np.max())
    # BIG_M_D: big-M for day-boundary constraints (C6). Only needs ≥ max_d.
    BIG_M_D = float(max_d)
    EPS     = 1e-6   # strict day-boundary upper bound (floor is left-closed)

    # Pre-compute deterministic reward matrices for each day offset (no Bernoulli)
    R_det = {}
    for k in k_set:
        day = min(start_day_idx + k, max_day)
        rm, _ = build_day_matrices(
            rate_stack[day], loads_stack[day], distance_arr, diesel_arr
        )
        R_det[k] = rm.to_numpy()   # shape (num_n, num_n), same as reward_matrix

    # ── Build PuLP model ─────────────────────────────────────────────────────
    prob = pulp.LpProblem(f"MIP_Exact_{start_node}", pulp.LpMaximize)

    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
         for i in nodes for j in nodes if i != j}

    # Arrival time at node i — exact when tight bounds are active on the route
    t = {i: pulp.LpVariable(f"t_{i}", lowBound=0.0, upBound=float(max_d))
         for i in nodes}

    # Day-arc indicator: z[i,j,k]=1 iff arc used AND node i in day slot k
    z = {(i, j, k): pulp.LpVariable(f"z_{i}_{j}_{k}", cat=pulp.LpBinary)
         for i in nodes for j in nodes if i != j
         for k in k_set}

    # Objective: day-exact deterministic reward
    prob += pulp.lpSum(
        float(R_det[k][i, j]) * z[(i, j, k)]
        for i in nodes for j in nodes if i != j
        for k in k_set
    )

    # C1 – Flow conservation + degree
    prob += pulp.lpSum(x[(start_node, j)] for j in other_nodes) == 1
    prob += pulp.lpSum(x[(j, start_node)] for j in other_nodes) == 1
    for i in other_nodes:
        prob += (pulp.lpSum(x[(i, j)] for j in nodes if j != i) ==
                 pulp.lpSum(x[(j, i)] for j in nodes if j != i))
        prob += pulp.lpSum(x[(j, i)] for j in nodes if j != i) <= 1

    # C2 – Duration
    prob += (pulp.lpSum(
        float(time_matrix_np[i][j]) * x[(i, j)]
        for i in nodes for j in nodes if i != j
    ) <= max_d)

    # C3 – Departure time at start node is always zero (day slot 0)
    prob += t[start_node] == 0.0

    # C4 – Exact arrival times (subtour elimination + day tracking).
    # Both bounds together force t[j] = t[i] + time[i,j] on every active arc.
    # BIG_M_T = max_d + max(time) ensures inactive arcs impose no spurious
    # lower bounds on arrival times of non-visited nodes.
    for j in other_nodes:
        for i in nodes:
            if i == j:
                continue
            tij = float(time_matrix_np[i][j])
            prob += t[j] >= t[i] + tij - BIG_M_T * (1 - x[(i, j)])   # lower
            prob += t[j] <= t[i] + tij + BIG_M_T * (1 - x[(i, j)])   # upper

    # C5 – Each arc is assigned to exactly one day slot
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            prob += pulp.lpSum(z[(i, j, k)] for k in k_set) == x[(i, j)]

    # C6 – Day-slot boundaries: z[i,j,k]=1 ↔ t[i] ∈ [14k, 14(k+1))
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            for k in k_set:
                prob += t[i] >= 14.0 * k       - BIG_M_D * (1 - z[(i, j, k)])
                prob += t[i] <= 14.0 * (k + 1) - EPS + BIG_M_D * (1 - z[(i, j, k)])

    # ── Solve ────────────────────────────────────────────────────────────────
    pulp.PULP_CBC_CMD(msg=0, timeLimit=600).solve(prob)

    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        return status, None, -np.inf, np.inf

    # ── Route reconstruction ─────────────────────────────────────────────────
    try:
        cur   = start_node
        route = [start_node]
        for _ in range(num_n + 1):
            moved = False
            for j in nodes:
                if j == cur:
                    continue
                v = x[(cur, j)].varValue
                if v is not None and v > 0.99:
                    route.append(j)
                    cur = j
                    moved = True
                    break
            if cur == start_node and len(route) > 1:
                break
            if not moved:
                return 'Error_In_Route', None, -np.inf, np.inf
        if route[0] != start_node or route[-1] != start_node:
            return 'Error_In_Route', None, -np.inf, np.inf
    except Exception as exc:
        print(f"  [MIP-Exact] reconstruction error (start={start_node}): {exc}")
        return 'Error_Exception', None, -np.inf, np.inf

    # Final reward via simulate_route_reward with avail_prob_arr=None (no Bernoulli)
    # to guarantee bit-exact parity with the DRL deterministic evaluation.
    total_reward, total_duration = simulate_route_reward(
        route, start_node, start_day_idx,
        time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    return status, route, total_reward, total_duration


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
    from problem_data import build_day_matrices

    num_days = rate_stack.shape[0]

    def reward_matrix_for_time(t_elapsed):
        day_idx = min(start_day_idx + int(t_elapsed // 14), num_days - 1)
        _, rm_pen = build_day_matrices(
            rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
        )
        return rm_pen

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
        rm = reward_matrix_for_time(time_elapsed)

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

    # Close cycle (time only — reward not accumulated here).
    if route[-1] != start_node:
        return_time = time_m[current_node][start_node]
        if time_elapsed + return_time <= max_d:
            time_elapsed += return_time
            route.append(start_node)

    if route[-1] != start_node or len(route) < 2:
        return "Infeasible", route if len(route) > 1 else None, -np.inf, time_elapsed, False

    # Canonical evaluation — bit-exact with MIP-Exact and DRL Det.
    total_reward, total_duration = simulate_route_reward(
        route, start_node, start_day_idx,
        time_m, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    is_valid = total_duration <= max_d
    status = "Optimal" if is_valid else "Infeasible"
    return status, route, total_reward, total_duration, is_valid
