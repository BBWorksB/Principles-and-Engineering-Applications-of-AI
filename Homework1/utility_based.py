import gym
import heapq


# Couldn't run on my PC but on trying on someone the code run hence didnt upload the video

def state_to_pos(state):
    """Convert state index to (row, col) position."""
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
        return 0  # Fallback, shouldn't occur in valid paths

def build_graph(env):
    """Construct adjacency list representing valid moves between safe cells."""
    graph = {}
    for row in range(8):
        for col in range(8):
            cell = env.desc[row][col].decode('utf-8')
            if cell == 'H':
                continue  # Skip holes
            neighbors = []
            # Check up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adj_row, adj_col = row + dr, col + dc
                if 0 <= adj_row < 8 and 0 <= adj_col < 8:
                    adj_cell = env.desc[adj_row][adj_col].decode('utf-8')
                    if adj_cell != 'H':
                        neighbors.append((adj_row, adj_col))
            graph[(row, col)] = neighbors
    return graph

def dijkstra(graph, start, goal):
    """Find the shortest path from start to goal using Dijkstra's algorithm."""
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    heap = [(0, start)]
    
    while heap:
        current_dist, current_node = heapq.heappop(heap)
        if current_node == goal:
            break
        if current_dist > distances[current_node]:
            continue
        for neighbor in graph[current_node]:
            dist = current_dist + 1  # All edges have equal weight
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                previous_nodes[neighbor] = current_node
                heapq.heappush(heap, (dist, neighbor))
    
    # Reconstruct path
    path = []
    current = goal
    if previous_nodes.get(current) is None and current != start:
        return None  # No path exists
    while current is not None:
        path.append(current)
        current = previous_nodes.get(current)
    path.reverse()
    if not path or path[0] != start:
        return None
    return path

# Initialize environment
map=["SFFFHFFF", "FHFFFHFF", "FFFHFHFH", "HFFFHFFF","FFFFHHFF","HFFFFFFF","FFHHHFHF","FFFHFFFG"]
env = gym.make('FrozenLake-v1', desc=map, map_name="8x8", is_slippery=False, render_mode="human")

# Identify start and goal positions
start_pos, goal_pos = None, None
for row_idx in range(8):
    for col_idx in range(8):
        cell = env.desc[row_idx][col_idx].decode('utf-8')
        if cell == 'S':
            start_pos = (row_idx, col_idx)
        elif cell == 'G':
            goal_pos = (row_idx, col_idx)

# Build graph and compute shortest path
graph = build_graph(env)
shortest_path = dijkstra(graph, start_pos, goal_pos) if start_pos and goal_pos else None

# Generate action sequence from path
path_actions = []
if shortest_path and len(shortest_path) > 1:
    for i in range(len(shortest_path) - 1):
        current = shortest_path[i]
        next_node = shortest_path[i + 1]
        action = get_action(current[0], current[1], next_node[0], next_node[1])
        path_actions.append(action)

# Execute the path repeatedly
current_action_idx = 0
observation, info = env.reset()
for _ in range(1000):
    if current_action_idx < len(path_actions):
        action = path_actions[current_action_idx]
        current_action_idx += 1
    else:
        action = 0  # Default if no path exists
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        current_action_idx = 0  # Reset to start of path

env.close()