import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the coin data
coin_data = scipy.io.loadmat("coin_data.mat")["coin_data_list"].flatten()
print("Coin flips:", coin_data)

# Set hypotheses and initial uniform prior
hypotheses = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
num_hypo = len(hypotheses)
prior = np.ones(num_hypo) / num_hypo

# Initialize storage
posteriors = [prior.copy()]         # List to hold posterior at each step
predictive_probs = [np.sum(hypotheses * prior)]  # Initial predictive prob

# Bayesian updating per flip
for flip in coin_data:
    current_posterior = posteriors[-1].copy()

    # Update likelihood per hypothesis
    for i in range(num_hypo):
        if flip == 1:
            likelihood = hypotheses[i]
        else:
            likelihood = 1 - hypotheses[i]
        current_posterior[i] *= likelihood

    # Normalize posterior
    current_posterior /= np.sum(current_posterior)

    # Append updated posterior and predictive probability
    posteriors.append(current_posterior)
    predictive_probs.append(np.sum(current_posterior * hypotheses))

# Convert results to arrays
posteriors = np.array(posteriors)
predictive_probs = np.array(predictive_probs)

# --- PLOTS ---

# Plot posterior over time
plt.figure(figsize=(10, 6))
for i in range(num_hypo):
    plt.plot(posteriors[:, i], label=f'h={hypotheses[i]:.1f}')
plt.title("Posterior Probability of Each Hypothesis Over Time")
plt.xlabel("Number of Observations")
plt.ylabel("Posterior Probability")
plt.legend()
plt.grid(True)
plt.show()

# Plot predictive probability
plt.figure(figsize=(10, 6))
plt.plot(predictive_probs, marker='o', markersize=3)
plt.title("Predictive Probability that Next Flip is Heads")
plt.xlabel("Number of Observations")
plt.ylabel("Predictive Probability")
plt.grid(True)
plt.show()

# Final inference
final_posterior = posteriors[-1]
most_likely_index = np.argmax(final_posterior)
print(f"\nMost likely hypothesis after all observations: h={hypotheses[most_likely_index]:.1f}")
