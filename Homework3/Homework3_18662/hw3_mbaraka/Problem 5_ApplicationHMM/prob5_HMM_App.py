import numpy as np

def forward_backward(observations, transition, emission, prior):
    T = len(observations)
    N = transition.shape[0]  # Number of hidden states (10)
    
    # Forward pass
    alpha = np.zeros((T, N))
    alpha[0] = prior * emission[:, observations[0]]
    alpha[0] /= np.sum(alpha[0])
    
    for t in range(1, T):
        alpha[t] = emission[:, observations[t]] * np.dot(alpha[t-1], transition)
        alpha[t] /= np.sum(alpha[t])
    
    # Backward pass
    beta = np.ones((T, N))
    for t in range(T-2, -1, -1):
        beta[t] = np.dot(transition, emission[:, observations[t+1]] * beta[t+1])
        beta[t] /= np.sum(beta[t])
    
    # Posterior probabilities
    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    
    return gamma

# Example usage
prior = np.ones(10) / 10  # Uniform prior

# Transition matrix (10x10)
transition = np.zeros((10, 10))
for i in range(10):
    transition[i, i] = 0.6
    transition[i, (i+1)%10] = 0.2
    transition[i, (i-1)%10] = 0.2

# Emission matrix (10x5)
emission = np.zeros((10, 5))
for i in range(10):
    emission[i, i%5] = 0.7
    emission[i, np.arange(5) != i%5] = 0.075

# Generating synthetic observations for testing
observations = np.random.choice(5, size=12, p=[0.7, 0.075, 0.075, 0.075, 0.075])

gamma = forward_backward(observations, transition, emission, prior)
most_likely_states = np.argmax(gamma, axis=1)

print("Most likely hidden states:", most_likely_states)