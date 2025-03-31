import gym
import numpy as np

# 4x4 map
map_4x4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

GAMMA = 0.9  # Discount factor
THETA = 1e-6  # Convergence threshold

def build_transition_model(desc):
    rows, cols = len(desc), len(desc[0])
    states = []
    goal = None
    holes = set()
    
    # Identify valid states (S and F)
    for r in range(rows):
        for c in range(cols):
            cell = desc[r][c]
            if cell in ['S', 'F']:
                states.append((r, c))
            elif cell == 'G':
                goal = (r, c)
            elif cell == 'H':
                holes.add((r, c))
    
    transitions = {}
    for s in states:
        transitions[s] = {}
        r, c = s
        
        for action in [0, 1, 2, 3]:
            # Define intended direction and raw probabilities (sum=0.9)
            if action == 0:  # LEFT
                dirs = [
                    (0, -1, 0.3),  # LEFT
                    (1, 0, 0.3),   # DOWN
                    (-1, 0, 0.2),   # UP
                    (0, 1, 0.1)     # RIGHT
                ]
            elif action == 1:  # DOWN
                dirs = [
                    (1, 0, 0.3),    # DOWN
                    (0, -1, 0.3),   # LEFT
                    (0, 1, 0.2),     # RIGHT
                    (-1, 0, 0.1)     # UP
                ]
            elif action == 2:  # RIGHT
                dirs = [
                    (0, 1, 0.3),    # RIGHT
                    (1, 0, 0.3),    # DOWN
                    (-1, 0, 0.2),   # UP
                    (0, -1, 0.1)    # LEFT
                ]
            elif action == 3:  # UP
                dirs = [
                    (-1, 0, 0.3),   # UP
                    (0, 1, 0.3),    # RIGHT
                    (1, 0, 0.2),     # DOWN
                    (0, -1, 0.1)     # LEFT
                ]
            
            next_states = {}
            for dr, dc, raw_prob in dirs:
                nr, nc = r + dr, c + dc
                # Check boundaries and holes
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in holes:
                    nr, nc = r, c  # Stay if invalid
                next_state = (nr, nc)
                # Assign reward
                reward = 1 if next_state == goal else 0
                next_states[next_state] = next_states.get(next_state, 0) + raw_prob
            
            # Normalize probabilities to sum to 1.0
            total = sum(next_states.values())
            transitions[s][action] = [
                (next_s, (prob / total), reward)
                for next_s, prob in next_states.items()
            ]
    
    return transitions, goal, holes

def value_iteration(desc):
    transitions, goal, holes = build_transition_model(desc)
    states = list(transitions.keys())
    V = {s: 0 for s in states}
    policy = {s: 0 for s in states}
    delta = float('inf')
    
    while delta >= THETA:
        delta = 0
        V_new = V.copy()
        for s in states:
            max_value = -np.inf
            best_action = 0
            for a in transitions[s]:
                total = 0
                for next_s, trans_prob, reward in transitions[s][a]:
                    if next_s == goal or next_s in holes:
                        total += trans_prob * (reward + GAMMA * 0)  # Terminal state
                    else:
                        total += trans_prob * (reward + GAMMA * V[next_s])
                if total > max_value:
                    max_value = total
                    best_action = a
            V_new[s] = max_value
            policy[s] = best_action
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new.copy()
    return V, policy

def main():
    env = gym.make(
        'FrozenLake-v1',
        desc=map_4x4,
        is_slippery=False,
        render_mode="human",
        disable_env_checker=True
    )
    V, policy = value_iteration(map_4x4)
    print("Value Function:")
    for s in sorted(V):
        print(f"{s}: {V[s]:.2f}")
    print("\nPolicy:")
    for s in sorted(policy):
        print(f"{s}: {policy[s]}")
    
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        r, c = obs // 4, obs % 4  # Convert observation to grid coordinates
        action = policy.get((r, c), 0)
        obs, _, done, _, _ = env.step(action)
    env.close()

if __name__ == "__main__":
    main()