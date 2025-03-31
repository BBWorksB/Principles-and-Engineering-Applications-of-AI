import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import math

# Download stopwords if not available
# nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam_ham_dataset copy.csv")  # Ensure the file is present

# Extract relevant columns
emails = df[['text', 'label_num']]

# Define spam-related keywords
spam_keywords = [ 'Viagra', 'Cialis', 'Levitra', 'Vicodin', 'Xanax',
'Ambien', 'OxyContin', 'Percocet', 'Valium', 'Tramadol', 'Adderall',
'Ritalin', 'Modafinil', 'Phentermine', 'Ativan', 'Klonopin', 'Zoloft',
'Prozac', 'Paxil', 'Wellbutrin', 'Celexa', 'Lexapro', 'Effexor', 'Zyprexa',
'Abilify', 'Seroquel', 'Lamictal', 'Depakote', 'Lithium', 'Nigerian',
'Lottery', 'Free', 'Discount', 'Limited time offer', 'Money-back guarantee',
'Investment opportunity', 'Secret', 'Unsubscribe', 'Click here', 'Double your money',
'Cash', 'Urgent', 'Get rich quick', 'Work from home', 'Multi-level marketing', 'Pyramid scheme', 'Enlarge', 'Buy now', 'Online pharmacy',
'Weight loss', 'Casino', 'Credit report', 'Debt relief']

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Compute prior probabilities
p_spam = len(df[df['label_num'] == 1]) / len(df)
p_ham = 1 - p_spam

# Compute likelihoods
def compute_likelihoods(emails, keywords):
    spam_counts = defaultdict(int)
    ham_counts = defaultdict(int)
    spam_emails = emails[emails['label_num'] == 1]
    ham_emails = emails[emails['label_num'] == 0]
    
    for _, row in spam_emails.iterrows():
        words = preprocess_text(row['text'])
        for word in words:
            if word in keywords:
                spam_counts[word] += 1
    
    for _, row in ham_emails.iterrows():
        words = preprocess_text(row['text'])
        for word in words:
            if word in keywords:
                ham_counts[word] += 1
    
    total_spam = len(spam_emails)
    total_ham = len(ham_emails)
    
    likelihoods_spam = {word: (spam_counts[word] + 1) / (total_spam + 2) for word in keywords}
    likelihoods_ham = {word: (ham_counts[word] + 1) / (total_ham + 2) for word in keywords}
    
    return likelihoods_spam, likelihoods_ham

likelihoods_spam, likelihoods_ham = compute_likelihoods(emails, spam_keywords)

# Bayesian Inference for Spam Classification
def classify_email(email_text):
    words = preprocess_text(email_text)
    log_p_spam = math.log(p_spam)
    log_p_ham = math.log(p_ham)
    
    for word in words:
        if word in spam_keywords:
            log_p_spam += math.log(likelihoods_spam.get(word, 1 / (len(emails) + 2)))
            log_p_ham += math.log(likelihoods_ham.get(word, 1 / (len(emails) + 2)))
    
    p_spam_given_email = math.exp(log_p_spam) / (math.exp(log_p_spam) + math.exp(log_p_ham))
    
    return p_spam_given_email

# Test the classifier
test_email = "needs Immediate action/attention"
print(f"Spam Probability: {classify_email(test_email)}")
