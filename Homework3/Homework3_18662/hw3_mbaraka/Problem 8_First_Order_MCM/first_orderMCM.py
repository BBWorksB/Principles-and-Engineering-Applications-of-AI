import numpy as np

# Found the customer churning dataset but later realised it is not of relevance to the dataset in the last minute had to generate 
# Synthtic data, the dataset is in the folder for reference.

# Generatic Synthetic Data
def generate_synthetic_sequences(num_customers=1000, max_sequence_length=20):
    """Generate synthetic purchase sequences using a Markov Chain."""
    np.random.seed(42)
    transition_matrix = np.array([[0.7, 0.3],  # From 0 to 0 and 0 to 1
                                  [0.4, 0.6]]) # From 1 to 0 and 1 to 1
    
    sequences = []
    for _ in range(num_customers):
        sequence = []
        current_state = np.random.choice([0, 1], p=[0.5, 0.5])  # Initial state
        sequence.append(current_state)
        for _ in range(max_sequence_length - 1):
            current_state = np.random.choice(
                [0, 1], 
                p=transition_matrix[current_state]
            )
            sequence.append(current_state)
        sequences.append(sequence)
    return sequences

sequences = generate_synthetic_sequences()


# Estimating the Transition Probabilities
def estimate_transition_probs(sequences):
    """Estimate initial and transition probabilities from data."""
    initial_counts = np.zeros(2)
    transition_counts = np.zeros((2, 2))
    
    for seq in sequences:
        initial_counts[seq[0]] += 1
        for i in range(len(seq) - 1):
            current = seq[i]
            next_state = seq[i+1]
            transition_counts[current][next_state] += 1
    
    # Normalize counts to probabilities
    initial_probs = initial_counts / initial_counts.sum()
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    
    return initial_probs, transition_probs

initial_probs, transition_probs = estimate_transition_probs(sequences)

# The Forward Algorithm (Computing the Alpha)
def forward_algorithm(sequence, initial_probs, transition_probs):
    """Compute forward probabilities (alpha) for a sequence."""
    T = len(sequence)
    alpha = np.zeros((T, 2))
    
    # Initialize alpha for the first state
    # Made an assumption that the first state is observed
    alpha[0][sequence[0]] = 1.0  
    
    for t in range(1, T):
        prev_state = sequence[t-1]
        curr_state = sequence[t]
        alpha[t][curr_state] = alpha[t-1][prev_state] * transition_probs[prev_state][curr_state]
    
    return alpha

# Testing the computing alpha
alpha = forward_algorithm(sequences[0], initial_probs, transition_probs)


# Backward Algorithm (Computing the Beta)
def backward_algorithm(sequence, transition_probs):
    """Compute backward probabilities (beta) for a sequence."""
    T = len(sequence)
    beta = np.zeros((T, 2))
    
    # Initialize beta for the last state
    # Just as above made the assumption that the first state was observed
    beta[T-1][sequence[T-1]] = 1.0  
    
    for t in range(T-2, -1, -1):
        curr_state = sequence[t]
        next_state = sequence[t+1]
        beta[t][curr_state] = transition_probs[curr_state][next_state] * beta[t+1][next_state]
    
    return beta

# testing the computation of Betta
beta = backward_algorithm(sequences[0], transition_probs)


# Posterior Probabilities (calculating Gamma)
def compute_posteriors(alpha, beta):
    """Compute posterior probabilities (gamma) for each state at each time."""
    T = alpha.shape[0]
    gamma = np.zeros((T, 2))
    
    # Total probability of the sequence (use alpha at the last time step)
    total_prob = alpha[-1][sequences[0][-1]]
    
    for t in range(T):
        # Element-wise multiplication
        gamma[t] = (alpha[t] * beta[t]) / total_prob  
        
    return gamma

# computing gamma
gamma = compute_posteriors(alpha, beta)


# Prediction Function
def predict_probability(history, transition_probs):
    """Predict the probability of making a purchase at the next time step."""
    last_state = history[-1]

    # Probability of transitioning to state 1 (purchase)
    return transition_probs[last_state][1]  

# Sample to test the model
sample_history = [1, 1, 1, 1, 0, 1]
print(f"Predicted purchase probability: {predict_probability(sample_history, transition_probs):.2f}")