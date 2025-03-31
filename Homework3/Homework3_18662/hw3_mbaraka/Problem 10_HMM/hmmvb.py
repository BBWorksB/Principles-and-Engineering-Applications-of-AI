import numpy as np

# Define the HMM parameters
states = ["Viral", "Bacterial"]  # Hidden states: 0 = Viral, 1 = Bacterial
observations = ["Normal", "High"]  # Observations: 0 = Normal, 1 = High
obs_sequence = ["Normal", "High", "High"]  # Observed sequence: N, H, H

# Initial probabilities for day 1
pi = np.array([0.7, 0.3])  # [P(Viral), P(Bacterial)] 

# Transition matrix (row = from state, column = to state) 
# Viral -> Viral, Viral -> Bacterial
# Bacterial -> Viral, Bacterial -> Bacterial
A = np.array([[0.8, 0.2],  # P(Viral -> Viral), P(Viral -> Bacterial)
              [0.4, 0.6]]) # P(Bacterial -> Viral), P(Bacterial -> Bacterial)


# Emission matrix (row = state, column = observation) 
# Viral -> Normal, Viral -> High
# Bacterial -> Normal, Bacterial -> High
B = np.array([[0.6, 0.4],  # P(Viral -> Normal), P(Viral -> High)
              [0.3, 0.7]]) # P(Bacterial -> Normal), P(Bacterial -> High)

# Convert observation sequence to indices
obs_indices = [0, 1, 1]  # Normal = 0, High = 1, High = 1 

# --- Forward Algorithm --- 

# Initialize the forward table: alpha[state][time]
alpha = np.zeros((2, 3))  # 2 states, 3 time steps

# Step 1: Initialization (t=0, observation = Normal)
# alpha[s][0] = pi[s] * B[s, obs[0]]
# TODO: initialize alpha for t = 0 
alpha[:, 0] = pi * B[:, obs_indices[0]]

# Step 2: Iterative Forward (t=1 and t=2, observations = High, High)
# TODO: implement the forward algorithm ß
for t in range(1, 3):
    for s in range(2):
        alpha[s, t] = np.sum(alpha[:, t-1] * A[:, s]) * B[s, obs_indices[t]]

# Step 3: Termination - sum the probabilities at the final time step 
# TODO: compute the total probability of the observation sequence ß
total_prob =  np.sum(alpha[:, 2]) # P(O) = alpha[Viral][2] + alpha[Bacterial][2] 

# --- Backward Algorithm ---
beta = np.zeros((2, 3))  # 2 states, 3 time steps

# Initialization (t=2, last time step)
# TODO: initialize beta for t = 2 
beta[:, 2] = 1

# Backward iteration (t=1 and t=0)
# TODO: implement the backward algorithm 
for t in range(1, -1, -1):
    for s in range(2):
        beta[s, t] = np.sum(A[s, :] * B[:, obs_indices[t+1]] * beta[:, t+1])

# Compute total probability using backward: P(O | model) = sum_s [pi[s] * B[s, obs[0]] * beta[s, 0]]
backward_prob = np.sum(pi * B[:, obs_indices[0]] * beta[:, 0]) # TODO: compute the backward probability 

# Print intermediate steps for clarity
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
