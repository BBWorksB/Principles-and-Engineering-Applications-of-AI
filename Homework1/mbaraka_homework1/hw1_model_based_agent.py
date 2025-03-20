# import gymnasium as gym
import gym
import numpy as np
from collections import deque

# Had to use gymnasium for it to run on my PC but on other PC had to use just gym
# Incase of an error comment out the gymnasium

# Initialize the environment with a specific map
map = ["SFFFHFFF", "FHFFFHFF", "FFFHFHFH", "HFFFHFFF",
       "FFFFHHFF", "HFFFFFFF", "FFHHHFHF", "FFFHFFFF"]
env = gym.make('FrozenLake-v1', desc=map, map_name="8x8",
               is_slippery=False, render_mode="human")

# Global variables to track the agent's knowledge
internal_map = [['U' for _ in range(8)] for _ in range(8)]  # 'U' for unknown
goal_pos = None  # Track the position of the goal once found

def state_to_pos(state):
    """Convert state index to (row, col)."""
    return state // 8, state % 8

def get_action(current_row, current_col, next_row, next_col):
    """Determine the action to move from current to next cell."""
    dr = next_row - current_row
    dc = next_col - current_col
    if dr == 1:
        return 1  # Down
    elif dr == -1:
        return 3  # Up
    elif dc == 1:
        return 2  # Right
    elif dc == -1:
        return 0  # Left
    else:
        return np.random.randint(0, 4)  # Fallback

def bfs_explore(start_row, start_col):
    """BFS to find the nearest cell adjacent to an unexplored (U) cell."""
    queue = deque([(start_row, start_col)])
    visited = {(start_row, start_col): None}
    found = None

    while queue:
        row, col = queue.popleft()

        # Check adjacent cells for unexplored
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_row, adj_col = row + dr, col + dc
            if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                if internal_map[adj_row][adj_col] == 'U':
                    found = (row, col)
                    break
        if found:
            break

        # Expand to neighboring safe cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row, next_col = row + dr, col + dc
            if 0 <= next_row < 8 and 0 <= next_col < 8:
                if (next_row, next_col) not in visited:
                    cell_status = internal_map[next_row][next_col]
                    if cell_status in ['F', 'S', 'G']:
                        visited[(next_row, next_col)] = (row, col)
                        queue.append((next_row, next_col))

    if not found:
        return None

    # Reconstruct path
    path = []
    current = found
    while current is not None:
        path.append(current)
        current = visited.get(current)
    path.reverse()
    print("len: ", len(path))
    return path[1:] if len(path) > 1 else None

def bfs_exploit(start_row, start_col, goal_pos):
    """BFS to find the shortest path to the goal, avoiding holes."""
    print("Ok ok ok ok")
    target_row, target_col = goal_pos
    queue = deque([(start_row, start_col)])
    visited = {(start_row, start_col): None}
    found = False

    while queue and not found:
        row, col = queue.popleft()
        if (row, col) == (target_row, target_col):
            found = True
            break

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row, next_col = row + dr, col + dc
            if 0 <= next_row < 8 and 0 <= next_col < 8:
                if (next_row, next_col) not in visited:
                    cell_status = internal_map[next_row][next_col]
                    if cell_status in ['F', 'S', 'G']:
                        visited[(next_row, next_col)] = (row, col)
                        queue.append((next_row, next_col))

    if not found:
        return None

    # Reconstruct path
    path = []
    current = (row, col)
    while current is not None:
        path.append(current)
        current = visited.get(current)
    path.reverse()
    return path[1:] if len(path) > 1 else None

def best_next_action(observation):
    """Determine the next action based on exploration or exploitation."""
    global goal_pos
    current_row, current_col = state_to_pos(observation)

    # Check if all cells are explored
    is_fully_explored = not any('U' in row for row in internal_map)

    if is_fully_explored:
        # Exploitation: navigate to goal
        if goal_pos is None:
            save_positions = [(r, c) for r in range(len(internal_map)) for c in range(len(internal_map[0])) if internal_map[r][c] == 'S']
            goal_pos = save_positions[np.random.randint(len(save_positions))]
            print("goood")
            # return np.random.randint(0, 4)  # Goal not found yet
        path = bfs_exploit(current_row, current_col, goal_pos)
        if path:
            next_row, next_col = path[0]
            return get_action(current_row, current_col, next_row, next_col)
        else:
            print("0000000")
            return np.random.randint(0, 4)  # No path found
    else:
        print("12344")
        # Exploration: check adjacent cells first
        adjacent_actions = []
        for dr, dc, action in [(-1, 0, 3), (1, 0, 1), (0, -1, 0), (0, 1, 2)]:
            adj_row, adj_col = current_row + dr, current_col + dc
            if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                if internal_map[adj_row][adj_col] == 'U':
                    adjacent_actions.append(action)
        if adjacent_actions:
            print("testing explot")
            return np.random.choice(adjacent_actions)
        else:
            # Use BFS to find the nearest U cell
            path = bfs_explore(current_row, current_col)
            if path:
                next_row, next_col = path[0]
                return get_action(current_row, current_col, next_row, next_col)
            else:
                return np.random.randint(0, 4)  # Fallback

# Initialize the environment and track the start cell
observation, info = env.reset()
start_row, start_col = state_to_pos(observation)
if internal_map[start_row][start_col] == 'U':
    internal_map[start_row][start_col] = 'S'

# Main interaction loop
for _ in range(1000):
    action = best_next_action(observation)
    new_observation, reward, terminated, truncated, info = env.step(action)

    # Update internal map based on the new state
    new_row, new_col = state_to_pos(new_observation)
    if terminated:
        if reward == 1:
            internal_map[new_row][new_col] = 'G'
            goal_pos = (new_row, new_col)
        else:
            internal_map[new_row][new_col] = 'H'
    else:
        if internal_map[new_row][new_col] == 'U':
            internal_map[new_row][new_col] = 'F'

    # Reset environment if terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()
        start_row, start_col = state_to_pos(observation)
        if internal_map[start_row][start_col] == 'U':
            internal_map[start_row][start_col] = 'S'
    else:
        observation = new_observation

env.close()