import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical # For sampling actions
import matplotlib.pyplot as plt
import pulp 
import time


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

    # 2b. Maximum steps constraint (total number of arcs used <= 5)
    prob += pulp.lpSum(x[i][j] for i in nodes for j in nodes if i != j) <= 5

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
    """
    Solves using a simple greedy heuristic:
    1. Always move to highest reward available node
    2. Return to start node when no more valid moves
    """
    current_node = start_node
    time_elapsed = 0.0
    total_reward = 0.0
    route = [start_node]
    visited = {start_node}
    return_threshold = 0.75 * max_d
    steps = 0
    while True and steps <5:  # Continue until we break
        # Find next highest reward move
        best_reward = -np.inf
        best_next_node = None
        
        for next_node in range(num_n):
            if next_node != current_node and next_node not in visited:
                step_time = time_m[current_node][next_node]
                return_time = time_m[next_node][start_node]  # Time to get back home
                step_reward = reward_m[current_node][next_node]
                
                # Check if we can make this move and still get back home
                total_future_time = time_elapsed + step_time + return_time
                if total_future_time <= max_d:
                    if step_reward > best_reward:
                        best_reward = step_reward
                        best_next_node = next_node

        if best_next_node is not None:
            # Make the move to the highest reward node
            time_elapsed += time_m[current_node][best_next_node]
            total_reward += reward_m[current_node][best_next_node]
            current_node = best_next_node
            route.append(current_node)
            visited.add(current_node)
            steps += 1
        else:
            # No more valid moves, return home
            return_time = time_m[current_node][start_node]
            if current_node != start_node:  # Only if we're not already home
                time_elapsed += return_time
                total_reward += reward_m[current_node][start_node]
                route.append(start_node)
                steps += 1
            break
        # If we've reached the time threshold, try to return home
        if time_elapsed >= return_threshold:
            return_time = time_m[current_node][start_node]
            time_elapsed += return_time
            total_reward += reward_m[current_node][start_node]
            route.append(start_node)
            steps += 1
            break
    if len(route) == 6 and route[-1] != start_node:
        # Force return home if we have 5 steps but not yet home
        #Change the last step to return home
        node_to_delete = route[-1]
        route = route[0:-1]
        current_node = route[-1]
        time_to_delete = time_m[current_node][node_to_delete]
        time_elapsed = time_elapsed - time_to_delete
        reward_to_delete = reward_m[current_node][node_to_delete]
        total_reward = total_reward - reward_to_delete
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time
        total_reward += reward_m[current_node][start_node]
    elif len(route) < 6 and route[-1] != start_node:
        current_node = route[-1]
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time
        total_reward += reward_m[current_node][start_node]
    # Final validation
    is_valid = False
    status = "Infeasible"
    
    if route[-1] == start_node and len(route) > 1:
        if  time_elapsed <= max_d:
            status = "Optimal"
            is_valid = True
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


def solve_2opt_heuristic(start_node, time_m, reward_m, max_d, num_n):
    """
    2-opt metaheuristic: Improves an initial feasible route by swapping two edges at a time.
    """
    # --- Step 1: Get initial feasible route (use greedy heuristic) ---
    status, route, total_reward, total_time, is_valid = solve_heuristic(
        start_node, time_m, reward_m, max_d, num_n
    )
    if not is_valid: 
        return status, route, total_reward, total_time, is_valid

    best_route = route[:]
    best_reward = total_reward
    best_time = total_time
    improved = True

    while improved:
        improved = False
        # Try all possible 2-opt swaps (skip start/end node)
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                # Create new route by reversing segment between i and j
                new_route = (
                    best_route[:i] +
                    best_route[i:j+1][::-1] +
                    best_route[j+1:]
                )
                # Compute total time and reward for new route
                new_time = 0.0
                new_reward = 0.0
                for k in range(len(new_route) - 1):
                    new_time += time_m[new_route[k]][new_route[k+1]]
                    new_reward += reward_m[new_route[k]][new_route[k+1]]
                # Check feasibility
                if new_time <= max_d and new_route[0] == start_node and new_route[-1] == start_node:
                    if new_reward > best_reward:
                        best_route = new_route
                        best_reward = new_reward
                        best_time = new_time
                        improved = True
                        break  # Accept first improvement (first-improvement strategy)
            if improved:
                break

    is_valid = best_time <= max_d and best_route[0] == start_node and best_route[-1] == start_node
    status = "Optimal" if is_valid else "Infeasible"
    return status, best_route, best_reward, best_time, is_valid


def solve_LNS_metaheuristic(start_node, time_m, reward_m, max_d, num_n):
    """
    Large Neighborhood Search (LNS) metaheuristic:
    1. Start with a greedy initial solution
    2. Iteratively destroy and repair neighborhoods
    3. Accept solutions if they improve best known or pass probabilistic criterion
    Returns: status, route, total_reward, total_duration (same format as solve_mip)
    """
    
    # --- Step 1: Generate initial solution using greedy heuristic ---
    status, init_route, init_reward, init_time, is_valid = solve_heuristic(
        start_node, time_m, reward_m, max_d, num_n
    )
    
    if not is_valid:
        return "Infeasible", None, -np.inf, np.inf
    
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
                          len(repaired_route) - 1 <= 6)  # Max 6 steps constraint
            
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
                len(best_route) - 1 <= 6)
    
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

def solve_genetic_algorithm(start_node, time_m, reward_m, max_d, num_n):
    """
    Hybrid Genetic Algorithm for routing:
    1. Initialize population with diverse feasible routes (max 6 nodes: 5 steps)
    2. Apply crossover, mutation (2-opt), and local search operators
    3. Return best solution from population
    Returns: status, route, total_reward, total_duration (same format as solve_mip)
    """
    
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
    if is_valid and len(route) <= 6:
        population.append({'route': route, 'reward': reward, 'duration': duration})
    # Seed 2: Nearest neighbor with randomization
    for _ in range(max(2, population_size // 4)):
        route = _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= 6:
            duration = sum(time_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            reward = sum(reward_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            if duration <= max_d:
                population.append({'route': route, 'reward': reward, 'duration': duration})
    # Seed 3: Random feasible routes
    for _ in range(max(2, population_size // 4)):
        route = _generate_random_feasible_route(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= 6:
            duration = sum(time_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            reward = sum(reward_m[route[i]][route[i+1]] for i in range(len(route) - 1))
            population.append({'route': route, 'reward': reward, 'duration': duration})
    # Fill remaining population slots
    while len(population) < population_size:
        route = _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n)
        if route and len(route) <= 6:
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
                    parent1['route'], parent2['route'], start_node, time_m, reward_m, max_d
                )
                
                if child_route and len(child_route) <= 6:
                    # Apply mutation (2-opt improvement)
                    if np.random.rand() < mutation_rate:
                        child_route = _apply_2opt_mutation(child_route, start_node, time_m, reward_m, max_d)
                    
                    # Evaluate child
                    child_duration = sum(time_m[child_route[i]][child_route[i+1]] for i in range(len(child_route) - 1))
                    child_reward = sum(reward_m[child_route[i]][child_route[i+1]] for i in range(len(child_route) - 1))
                    
                    if child_duration <= max_d and len(child_route) <= 6:
                        offspring.append({'route': child_route, 'reward': child_reward, 'duration': child_duration})
            else:
                # Pure mutation (2-opt on existing solution)
                parent = new_population[np.random.randint(0, len(new_population))]
                mutant_route = _apply_2opt_mutation(parent['route'][:], start_node, time_m, reward_m, max_d)
                
                if mutant_route and len(mutant_route) <= 6:
                    mutant_duration = sum(time_m[mutant_route[i]][mutant_route[i+1]] for i in range(len(mutant_route) - 1))
                    mutant_reward = sum(reward_m[mutant_route[i]][mutant_route[i+1]] for i in range(len(mutant_route) - 1))
                    
                    if mutant_duration <= max_d and len(mutant_route) <= 6:
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
                len(best_overall['route']) <= 6 and
                len(best_overall['route']) >= 2)
    
    final_status = "Optimal" if is_valid else "Infeasible"
    
    return final_status, best_overall['route'], best_overall['reward'], best_overall['duration']


def _generate_route_nearest_neighbor(start_node, time_m, reward_m, max_d, num_n):
    """
    Nearest neighbor heuristic with randomization:
    Start from a node and greedily move to closest unvisited node.
    Max 5 steps (6 nodes total including start/end)
    """
    current_node = start_node
    route = [start_node]
    visited = {start_node}
    time_elapsed = 0.0
    steps = 0
    max_steps = 5  # Max 5 steps = 6 nodes total
    
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
    
    if len(route) == 6 and current_node != start_node:
        # Force return home if we have 5 steps but not yet home
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
    elif len(route) < 6 and route[-1] != start_node:
        current_node = route[-1]
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time
    
    if time_elapsed <= max_d and route[0] == route[-1] == start_node and len(route) <= 6:
        return route
    return None


def _generate_random_feasible_route(start_node, time_m, reward_m, max_d, num_n):
    """
    Generate a random feasible route by randomly selecting nodes.
    Max 5 steps (6 nodes total including start/end)
    """
    current_node = start_node
    route = [start_node]
    visited = {start_node}
    time_elapsed = 0.0
    steps = 0
    max_steps = 5  # Max 5 steps = 6 nodes total
    
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
    if len(route) == 6 and current_node != start_node:
        # Force return home if we have 5 steps but not yet home
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
    elif len(route) < 6 and route[-1] != start_node:
        current_node = route[-1]
        route.append(start_node)
        #Now update time and reward accordingly
        return_time = time_m[current_node][start_node]
        time_elapsed += return_time
    
    if time_elapsed <= max_d and route[0] == route[-1] == start_node and len(route) <= 6:
        return route
    return None


def _crossover_routes(route1, route2, start_node, time_m, reward_m, max_d):
    """
    Order Crossover (OX): Combines two routes by:
    1. Copy segment from parent1
    2. Fill remaining nodes from parent2 in order
    Respects max 6 nodes (5 steps) constraint
    """
    if len(route1) < 3 or len(route2) < 3:
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
        if node not in child_nodes and len(child) < 5:  # Leave room for return to start
            child.append(node)
            child_nodes.add(node)
    
    # Close the route
    child.append(start_node)
    
    # Validate feasibility
    if len(child) > 6:
        child = child[:6]
        child[-1] = start_node
    
    duration = sum(time_m[child[i]][child[i+1]] for i in range(len(child) - 1))
    
    if duration <= max_d and len(child) <= 6 and len(child) >= 2:
        return child
    return None


def _apply_2opt_mutation(route, start_node, time_m, reward_m, max_d):
    """
    2-opt mutation: Try reversing a segment to improve the route.
    Maintains max 6 nodes constraint.
    """
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
                if len(new_route) > 6:
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
    
    return best_route if len(best_route) <= 6 else None