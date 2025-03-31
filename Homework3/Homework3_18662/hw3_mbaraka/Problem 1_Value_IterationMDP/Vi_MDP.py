import gym
import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Performs Value Iteration on a given FrozenLake environment.
    Returns the optimal value function and policy.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    V = np.zeros(num_states)  # Initialize value function to zeros
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        delta = 0  # Change in value function
        for s in range(num_states):
            if env.unwrapped.desc.flatten()[s] in [b'G', b'H']:
                continue  # Skip terminal states
            
            v = V[s]  # Store old value
            action_values = np.zeros(num_actions)
            
            for a in range(num_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
            
            V[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break  # Convergence
    
    return V, policy

def simulate_frozenlake(env, policy):
    """Simulates an episode using the computed optimal policy."""
    state, _ = env.reset()
    env.render()
    done = False
    
    while not done:
        action = policy[state]
        state, reward, done, truncated, info = env.step(action)
        # state, reward, terminated, truncated, info = env.step(action)
        # done = terminated or truncated

        env.render()
        if done:
            print(f"Episode finished with reward={reward}")
            break

def main():
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode='human')
    V, policy = value_iteration(env)
    print("Optimal Value Function:")
    print(V.reshape((4,4)))
    print("Optimal Policy:")
    print(policy.reshape((4,4)))
    
    simulate_frozenlake(env, policy)
    env.close()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    main()
