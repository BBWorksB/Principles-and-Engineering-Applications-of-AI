import numpy as np
import pandas as pd
import re
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopword and punctuations
# nltk.download('stopwords')
# nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    """
    Prepocess by:
    - lowercase
    - punctuation removing
    - tokenizing words
    -Romovind stop words    
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Training function for MAP-based classification
def train_map_classifier(documents, labels):
    """
    Trains the MAP-based classifier using Maximum A Posteriori estimation.
    
    This function:
    - Computes class priors P(C) based on the frequency of class occurrences.
    - Computes word likelihoods P(w | C) using Laplace smoothing to handle unseen words.
    
    Parameters:
    - documents: List of text documents.
    - labels: Corresponding class labels for each document.
    - alpha: Smoothing parameter (default is 1 for Laplace smoothing).
    
    Returns:
    - class_probs: Dictionary of class prior probabilities.
    - word_probs: Dictionary of word likelihoods per class.
    - vocabulary: Set of all words appearing in the training corpus.
    """
    # Initialize dictionaries to store class probabilities and word probabilities
    class_probs = defaultdict(float)
    word_probs = defaultdict(lambda: defaultdict(float))
    word_counts = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)
    vocabulary = set()
    alpha = 1  # Laplace smoothing parameter

    # Count occurrences of each class and each word in each class
    for doc, label in zip(documents, labels):
        class_counts[label] += 1
        tokens = preprocess_text(doc)
        for token in tokens:
            word_counts[label][token] += 1
            vocabulary.add(token)

    # Compute class prior probabilities P(C)
    total_docs = len(documents)
    for label, count in class_counts.items():
        class_probs[label] = count / total_docs

    # Compute word likelihoods P(w | C) with Laplace smoothing
    for label, words in word_counts.items():
        total_words_in_class = sum(words.values())
        for word in vocabulary:
            word_probs[label][word] = (words[word] + alpha) / (total_words_in_class + alpha * len(vocabulary))

    return class_probs, word_probs, vocabulary

# Function to predict the most probable class
def predict_map(document, class_probs, word_probs, vocabulary):
    """
    Predicts the most probable class for a given document using MAP estimation.
    
    This function computes:
    log(P(C)) + sum(log(P(w | C))) for all words in the document
    and selects the class with the highest computed probability.
    
    Parameters:
    - document: The input document to classify.
    - class_probs: Dictionary of class prior probabilities.
    - word_probs: Dictionary of word likelihoods per class.
    - vocabulary: Set of words in the training data.
    
    Returns:
    - The predicted class label.
    """
    # Preprocess the input document
    tokens = preprocess_text(document)
    
    # Initialize a dictionary to store log probabilities for each class
    class_log_probs = defaultdict(float)
    
    # Compute log probabilities for each class
    for label in class_probs:
        # Start with the log of the prior probability P(C)
        class_log_probs[label] = np.log(class_probs[label])
        
        # Add the log of the likelihoods P(w | C) for each word in the document
        for token in tokens:
            if token in vocabulary:
                class_log_probs[label] += np.log(word_probs[label].get(token, 1e-6))  # Use a small value for unseen words
    
    # Return the class with the highest log probability
    return max(class_log_probs, key=class_log_probs.get)
   

# Main execution block
if __name__ == "__main__":
    # Load dataset
    data = [
        ("The match was intense", "Sports"),
        ("The parliament passed a new law", "Politics"),
        ("A great game by the team", "Sports"),
        ("Political debate heats up", "Politics"),
        ("Football is a popular sport", "Sports"),
        ("The government announced new policies", "Politics"),
        ("Basketball players scored high", "Sports"),
        ("Election results were announced", "Politics"),
        ("Team wins the championship after a great match", "Sports"),
        ("Government passes a new economic policy", "Politics"),
        ("Player scores a goal in the final minute", "Sports"),
        ("New tax reforms announced by the president", "Politics")
    ]
    
    # Train-Test split
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # Train the MAP classifier
    class_probs, word_probs, vocabulary = train_map_classifier(
        [doc for doc, label in train_data], [label for doc, label in train_data]
    )

    # Test prediction on new sample
    # test_doc = "The championship match was intense and thrilling"
    test_doc ="The new laws are not supportive of the citizens"
    # test_doc = "So whats your name again?"
    predicted_class = predict_map(test_doc, class_probs, word_probs, vocabulary)
    
    print(f"Predicted class: {predicted_class}")
