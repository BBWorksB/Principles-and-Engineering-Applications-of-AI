import gym
import numpy as np
import random

def epsilon_greedy(Q, state, epsilon):
    """Chooses an action using the epsilon-greedy strategy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(Q.shape[1]))
    else:
        return np.argmax(Q[state, :])

def sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=10000):
    """
    Implements the SARSA algorithm for FrozenLake.
    Returns the optimal Q-table and policy.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    
    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        done = False
        
        while not done:
            next_state, reward, done, truncated, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
    
    policy = np.argmax(Q, axis=1)
    return Q, policy

def simulate_frozenlake_sarsa(env, policy):
    """Simulates an episode using the SARSA-trained policy."""
    state, _ = env.reset()
    env.render()
    done = False
    
    while not done:
        action = policy[state]
        state, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            print(f"Episode finished with reward={reward}")
            break

def main():
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode='human')
    Q, policy = sarsa(env)
    print("Optimal Q-Table:")
    print(Q.reshape((4,4,4)))
    print("Optimal Policy:")
    print(policy.reshape((4,4)))
    
    simulate_frozenlake_sarsa(env, policy)
    env.close()

if __name__ == "__main__":
    main()
