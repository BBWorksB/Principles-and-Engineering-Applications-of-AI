import pandas as pd
from fuzzywuzzy import fuzz
import string
from nltk.corpus import stopwords
import nltk

# nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]



def is_keyword_present(tokens, keyword, threshold=80):
    return any(fuzz.partial_ratio(token, keyword) >= threshold for token in tokens)


spam_keywords = [ 'Viagra', 'Cialis', 'Levitra', 'Vicodin', 'Xanax',
'Ambien', 'OxyContin', 'Percocet', 'Valium', 'Tramadol', 'Adderall',
'Ritalin', 'Modafinil', 'Phentermine', 'Ativan', 'Klonopin', 'Zoloft',
'Prozac', 'Paxil', 'Wellbutrin', 'Celexa', 'Lexapro', 'Effexor', 'Zyprexa',
'Abilify', 'Seroquel', 'Lamictal', 'Depakote', 'Lithium', 'Nigerian',
'Lottery', 'Free', 'Discount', 'Limited time offer', 'Money-back guarantee',
'Investment opportunity', 'Secret', 'Unsubscribe', 'Click here', 'Double your money',
'Cash', 'Urgent', 'Get rich quick', 'Work from home', 'Multi-level marketing', 'Pyramid scheme', 'Enlarge', 'Buy now', 'Online pharmacy',
'Weight loss', 'Casino', 'Credit report', 'Debt relief']

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')
emails = df['text'].tolist()
labels = df['label_num'].tolist()

# Compute keyword presence using fuzzy matching
presence_data = []
for email in emails:
    tokens = preprocess_text(email)
    presence_row = {keyword: is_keyword_present(tokens, keyword.lower()) for keyword in spam_keywords}
    presence_data.append(presence_row)
presence_df = pd.DataFrame(presence_data)
presence_df['label'] = labels

# Calculate CPTs with Laplace smoothing (alpha=1)
alpha = 1
total_spam = sum(labels)
total_ham = len(labels) - total_spam

cpds = []
prior_spam = total_spam / len(labels)


# Build Bayesian Network
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Create CPT for 'spam'
cpd_spam = TabularCPD(
    variable='spam',
    variable_card=2,
    values=[[1 - prior_spam], [prior_spam]],
    state_names={'spam': ['ham', 'spam']}
)

# Create CPTs for keywords
for keyword in spam_keywords:
    spam_present = presence_df[presence_df['label'] == 1][keyword].sum()
    ham_present = presence_df[presence_df['label'] == 0][keyword].sum()
    p_spam = (spam_present + alpha) / (total_spam + 2*alpha)
    p_ham = (ham_present + alpha) / (total_ham + 2*alpha)
    cpd = TabularCPD(
        variable=keyword,
        variable_card=2,
        evidence=['spam'],
        evidence_card=[2],
        values=[[1 - p_ham, 1 - p_spam], [p_ham, p_spam]],
        state_names={keyword: ['absent', 'present'], 'spam': ['ham', 'spam']}
    )
    cpds.append(cpd)

# Build model
model = BayesianNetwork([('spam', keyword) for keyword in spam_keywords])
model.add_cpds(cpd_spam, *cpds)
infer = VariableElimination(model)


# Predicting function
def predict_spam_probability(text):
    tokens = preprocess_text(text)
    evidence = {keyword: 'present' if is_keyword_present(tokens, keyword.lower()) else 'absent' for keyword in spam_keywords}
    prob = infer.query(variables=['spam'], evidence=evidence)
    return prob.values[1]  # P(spam='spam')


# Testing
test_email = "needs Immediate action/attention"
print(f"Spam Probability: {predict_spam_probability(test_email):.4f}")
