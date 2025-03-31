import numpy as np

# Define the HMM parameters
states = ["Viral", "Bacterial"]  # Hidden states: 0 = Viral, 1 = Bacterial
observations = ["Normal", "High"]  # Observations: 0 = Normal, 1 = High
obs_sequence = ["Normal", "High", "High"]  # Observed sequence: N, H, H

# Initial probabilities for day 1
pi = np.array([0.7, 0.3])  # [P(Viral), P(Bacterial)] 

# Transition matrix (row = from state, column = to state) 
A = np.array([[0.8, 0.2],  # P(Viral -> Viral), P(Viral -> Bacterial)
              [0.4, 0.6]]) # P(Bacterial -> Viral), P(Bacterial -> Bacterial)

# Emission matrix (row = state, column = observation) 
B = np.array([[0.6, 0.4],  # P(Viral -> Normal), P(Viral -> High)
              [0.3, 0.7]]) # P(Bacterial -> Normal), P(Bacterial -> High)

# Convert observation sequence to indices
obs_indices = [0, 1, 1]  # Normal = 0, High = 1, High = 1 

# --- Forward Algorithm --- 
alpha = np.zeros((2, 3))  # 2 states, 3 time steps

# Step 1: Initialization (t=0)
alpha[:, 0] = pi * B[:, obs_indices[0]]

# Step 2: Iterative Forward
for t in range(1, 3):
    for s in range(2):
        alpha[s, t] = np.sum(alpha[:, t-1] * A[:, s]) * B[s, obs_indices[t]]

# Step 3: Termination
total_prob = np.sum(alpha[:, 2])

# --- Backward Algorithm ---
beta = np.zeros((2, 3))  # 2 states, 3 time steps

# Step 1: Initialization (t=2)
beta[:, 2] = 1

# Step 2: Backward iteration
for t in range(1, -1, -1):
    for s in range(2):
        beta[s, t] = np.sum(A[s, :] * B[:, obs_indices[t+1]] * beta[:, t+1])

# Compute probability using backward algorithm
backward_prob = np.sum(pi * B[:, obs_indices[0]] * beta[:, 0])

# Print results
print("Forward Algorithm Table (alpha[state][time]):")
print(f"t=0 (Normal): Viral = {alpha[0, 0]:.4f}, Bacterial = {alpha[1, 0]:.4f}")
print(f"t=1 (High):   Viral = {alpha[0, 1]:.4f}, Bacterial = {alpha[1, 1]:.4f}")
print(f"t=2 (High):   Viral = {alpha[0, 2]:.4f}, Bacterial = {alpha[1, 2]:.4f}")
print(f"\nProbability of the observation sequence (N, H, H): {total_prob:.6f}") 

print("\nBackward Algorithm Table (beta[state][time]):")
print(f"t=2 (High):   Viral = {beta[0, 2]:.4f}, Bacterial = {beta[1, 2]:.4f}")
print(f"t=1 (High):   Viral = {beta[0, 1]:.4f}, Bacterial = {beta[1, 1]:.4f}")
print(f"t=0 (Normal): Viral = {beta[0, 0]:.4f}, Bacterial = {beta[1, 0]:.4f}")
print(f"Backward Probability: {backward_prob:.6f}")
