import gym
import numpy as np
from heapq import heappush, heappop

# Define the map structure (replaced during testing)
map = [
    "SFFFHFFF",
    "FHFFFHFF",
    "FFFHFHFH",
    "HFFFHFFF",
    "FFFFHHFF",
    "HFFFFFFF",
    "FFHHHFHF",
    "FFFHFFFG"
]

# Create the FrozenLake environment with the specified map
env = gym.make(
    'FrozenLake-v1',
    desc=map,
    map_name="8x8",
    is_slippery=False,     # ensures deterministic movement
    render_mode="human"    # shows the environment visually
)

# FrozenLake actions:
# 0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP
ACTIONS = [0, 1, 2, 3]
ACTION_VECTORS = {
    0: (0, -1),   # LEFT
    1: (1,  0),   # DOWN
    2: (0,  1),   # RIGHT
    3: (-1, 0)    # UP
}

class UtilityBasedAgent:
    """
    Utility-Based Agent that:
      1) Reads the map from the environment (fully observable).
      2) Constructs a graph using an adjacency list (safe squares only).
      3) Uses Dijkstra's algorithm to find the shortest path from S to G.
      4) Executes that path action-by-action.
    """
    def __init__(self):
        self.planned_actions = []

    def reset(self):
        """
        Clears out any old plan. We will re-compute the path each time
        we reset or run a new episode.
        """
        self.planned_actions = []

    def build_graph(self, desc):
        """
        Constructs a graph representation from the environment's map.

        Parameters:
        desc (list of strings): 8x8 list representation of the environment.

        Returns:
        graph (dict): Adjacency list where keys are (row, col) positions and values
                      are lists of reachable neighbors.
        start (tuple): The (row, col) position of 'S'.
        goal (tuple): The (row, col) position of 'G'.
        """
        rows, cols = len(desc), len(desc[0])
        graph = {}
        start, goal = None, None

        for r in range(rows):
            for c in range(cols):
                cell = desc[r][c]
                if cell == 'S':
                    start = (r, c)
                    graph[(r, c)] = []
                elif cell == 'G':
                    goal = (r, c)
                    graph[(r, c)] = []
                elif cell == 'F':
                    graph[(r, c)] = []
                # Holes ('H') are NOT added to the graph

        # Create edges (valid moves between safe squares)
        for (r, c) in graph:
            for action, (dr, dc) in ACTION_VECTORS.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) in graph:  # Ensure it's a valid safe move
                    graph[(r, c)].append(((nr, nc), action))  # (neighbor, action)

        return graph, start, goal

    def dijkstra_shortest_path(self, graph, start, goal):
        """
        Implements Dijkstra's algorithm to find the shortest path from start to goal.

        Parameters:
        graph (dict): Adjacency list representation of the environment.
        start (tuple): Starting position (row, col).
        goal (tuple): Goal position (row, col).

        Returns:
        path (list): A sequence of actions leading from start to goal.
        """
        if start == goal:
            return []

        # Priority queue for Dijkstra (distance, node)
        pq = []
        heappush(pq, (0, start))
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        parent = {}

        while pq:
            dist, current = heappop(pq)

            if current == goal:
                break

            for neighbor, action in graph[current]:
                new_dist = dist + 1  # All moves have equal cost
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    parent[neighbor] = (current, action)
                    heappush(pq, (new_dist, neighbor))

        # Reconstruct path from goal to start
        if goal not in parent:
            print("No path found!")
            return []

        path_actions = []
        node = goal
        while node != start:
            node, action = parent[node]
            path_actions.append(action)

        path_actions.reverse()  # Because we backtracked from the goal
        return path_actions

    def plan_shortest_path(self, desc):
        """
        Plans the shortest path using Dijkstra's algorithm.

        Parameters:
        desc (list of strings): The map description.

        Stores:
        self.planned_actions (list): The computed path of actions to follow.
        """
        graph, start, goal = self.build_graph(desc)
        self.planned_actions = self.dijkstra_shortest_path(graph, start, goal)

    def choose_action(self, obs):
        """
        Returns the next action from the planned path.

        Parameters:
        obs (int): The current observation (not directly used in Dijkstra’s algorithm).

        Returns:
        int: The next action (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP).
        """
        if self.planned_actions:
            return self.planned_actions.pop(0)
        return 0  # Default action if no planned path exists (should not happen).

def main():
    """
    Main driver code.
    1) Reset the environment and create an agent.
    2) Agent inspects the map to plan the shortest path from S to G.
    3) Agent follows that path in the environment.
    """
    obs, _ = env.reset()

    agent = UtilityBasedAgent()
    agent.reset()

    # Read the map from the environment
    desc = map  # This will change when tested with different maps

    # Compute shortest path using Dijkstra
    agent.plan_shortest_path(desc)

    done = False
    while not done:
        env.render()

        # Get the next move in the precomputed path
        action = agent.choose_action(obs)

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)

        if done:
            print(f"Episode finished with reward={reward}")
            break

    env.close()

if __name__ == "__main__":
    main()
