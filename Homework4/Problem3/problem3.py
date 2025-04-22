import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the coin data
coin_data = scipy.io.loadmat('coin_data.mat')['coin_data_list'].flatten()
thetas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Compute ML estimates
ml_estimates = []
heads_so_far = 0
for i, flip in enumerate(coin_data):
    heads_so_far += flip
    ml = heads_so_far / (i + 1)
    ml_estimates.append(ml)

# Load Bayesian and MAP predictions from Problem 1 and 2 (code reused)
# Bayesian predictions (from Problem 1)
prior_bayesian = np.ones(6) / 6
posteriors_bayesian = [prior_bayesian.copy()]
bayesian_pred = [np.sum(thetas * prior_bayesian)]
thetas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
for flip in coin_data:
    current_posterior = posteriors_bayesian[-1].copy()
    for i in range(len(thetas)):
        likelihood = thetas[i] if flip == 1 else (1 - thetas[i])
        current_posterior[i] *= likelihood
    current_posterior /= current_posterior.sum()
    posteriors_bayesian.append(current_posterior)
    bayesian_pred.append(np.sum(thetas * current_posterior))

# MAP predictions (from Problem 2, uniform prior)
def compute_map_predictions(prior, coin_data, thetas):
    posteriors = [prior.copy()]
    predictive_probs = [thetas[np.argmax(prior)]]
    for flip in coin_data:
        current_posterior = posteriors[-1].copy()
        for i in range(len(thetas)):
            likelihood = thetas[i] if flip == 1 else (1 - thetas[i])
            current_posterior[i] *= likelihood
        current_posterior /= current_posterior.sum()
        predictive_probs.append(thetas[np.argmax(current_posterior)])
    return predictive_probs

prior_uniform = np.ones(6) / 6
map_uniform_pred = compute_map_predictions(prior_uniform, coin_data, thetas)

# Plot all predictions
plt.figure(figsize=(12, 6))
plt.plot(bayesian_pred, label='Bayesian Learning (Uniform Prior)', linestyle='--', alpha=0.8)
plt.plot(map_uniform_pred, label='MAP (Uniform Prior)', marker='o', markersize=4)
plt.plot(range(1, len(coin_data)+1), ml_estimates, label='ML Estimate', marker='x', markersize=4)
plt.xlabel('Number of Observations')
plt.ylabel('Probability Next Flip is Heads')
plt.title('Predictive Probability: Bayesian vs MAP vs ML')
plt.legend()
plt.grid(True)
plt.show()


print("\nProblem 3 Results:")
print(f"Final ML estimate: {ml_estimates[-1]:.4f}")