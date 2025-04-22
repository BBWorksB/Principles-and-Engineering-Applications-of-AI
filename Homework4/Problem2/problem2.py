import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the coin data
coin_data = scipy.io.loadmat('coin_data.mat')['coin_data_list'].flatten()
thetas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Function to compute MAP predictions given a prior
def compute_map_predictions(prior, coin_data, thetas):
    num_hypotheses = len(thetas)
    posteriors = [prior.copy()]
    predictive_probs = [thetas[np.argmax(prior)]]  # Initial MAP prediction
    
    for flip in coin_data:
        current_posterior = posteriors[-1].copy()
        # Update posterior using likelihood
        for i in range(num_hypotheses):
            likelihood = thetas[i] if flip == 1 else (1 - thetas[i])
            current_posterior[i] *= likelihood
        # Normalize and store
        current_posterior /= current_posterior.sum()
        posteriors.append(current_posterior)
        # Get MAP prediction
        predictive_probs.append(thetas[np.argmax(current_posterior)])
    
    return predictive_probs

# Define priors
prior_uniform = np.ones(6) / 6
prior_non_uniform = np.array([0.1, 0.1, 0.3, 0.3, 0.1, 0.1])

# Compute predictions for MAP with both priors
map_uniform_pred = compute_map_predictions(prior_uniform, coin_data, thetas)
map_non_uniform_pred = compute_map_predictions(prior_non_uniform, coin_data, thetas)

# Load Bayesian predictions from Problem 1
prior_bayesian = np.ones(6) / 6
posteriors_bayesian = [prior_bayesian.copy()]
bayesian_pred = [np.sum(thetas * prior_bayesian)]
for flip in coin_data:
    current_posterior = posteriors_bayesian[-1].copy()
    for i in range(len(thetas)):
        likelihood = thetas[i] if flip == 1 else (1 - thetas[i])
        current_posterior[i] *= likelihood
    current_posterior /= current_posterior.sum()
    posteriors_bayesian.append(current_posterior)
    bayesian_pred.append(np.sum(thetas * current_posterior))

# Plot all predictions
plt.figure(figsize=(12, 6))
plt.plot(bayesian_pred, label='Bayesian Learning (Uniform Prior)', linestyle='--', alpha=0.8)
plt.plot(map_uniform_pred, label='MAP (Uniform Prior)', marker='o', markersize=4)
plt.plot(map_non_uniform_pred, label='MAP (Non-Uniform Prior)', marker='s', markersize=4)
plt.xlabel('Number of Observations')
plt.ylabel('Probability Next Flip is Heads')
plt.title('Comparison of Predictive Probabilities: Bayesian vs MAP')
plt.legend()
plt.grid(True)
plt.show()

# Print final predictions
print("\nProblem 2 Results:")
print(f"MAP final prediction (Uniform Prior): {map_uniform_pred[-1]:.4f}")
print(f"MAP final prediction (Non-Uniform Prior): {map_non_uniform_pred[-1]:.4f}")
