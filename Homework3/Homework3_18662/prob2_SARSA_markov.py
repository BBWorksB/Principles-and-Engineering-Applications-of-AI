import gym
import numpy as np

# 4x4 map
map_4x4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

GAMMA = 0.99    # Higher discount factor to prioritize future rewards
ALPHA = 0.5     # Increased learning rate for faster convergence
EPSILON = 0.1   # Exploration rate (as per problem specs)
EPISODES = 100000  # Increased episodes for sufficient exploration

def epsilon_greedy(Q, state, n_actions):
    if np.random.random() < EPSILON:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

def sarsa(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    for _ in range(EPISODES):
        obs, _ = env.reset()
        current_state = obs
        action = epsilon_greedy(Q, current_state, n_actions)
        done = False
        
        while not done:
            next_obs, reward, done, _, _ = env.step(action)
            next_state = next_obs
            
            if done:
                target = reward
            else:
                next_action = epsilon_greedy(Q, next_state, n_actions)
                target = reward + GAMMA * Q[next_state][next_action]
            
            # Update Q-value
            Q[current_state][action] += ALPHA * (target - Q[current_state][action])
            
            if not done:
                current_state = next_state
                action = next_action
    
    # Extract policy and value function
    policy = {s: np.argmax(Q[s]) for s in range(n_states)}
    V = {s: np.max(Q[s]) for s in range(n_states)}
    return V, policy

def main():
    env = gym.make(
        'FrozenLake-v1',
        desc=map_4x4,
        is_slippery=False,
        render_mode="human",
        disable_env_checker=True
    )
    V, policy = sarsa(env)
    
    print("Value Function (SARSA):")
    for s in sorted(V):
        r, c = s // 4, s % 4
        print(f"({r}, {c}): {V[s]:.2f}")
    
    print("\nPolicy (SARSA):")
    for s in sorted(policy):
        r, c = s // 4, s % 4
        print(f"({r}, {c}): {policy[s]}")
    
    # Test the policy
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action = policy[obs]
        obs, _, done, _, _ = env.step(action)
    env.close()

if __name__ == "__main__":
    main()