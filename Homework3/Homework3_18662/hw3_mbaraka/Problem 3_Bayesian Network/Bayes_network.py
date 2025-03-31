# Implement bayesian network in python to find the probability the patient has strep throat (you may use pgmpy library).

# Import libraries
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the Bayesian Network Structure this is by creating a Direct Acyclic Graph based on the dependencies given
model = BayesianNetwork([
    # The disease affects fever, cough and sore throut
    ("Disease", "Fever"), 
    ("Disease", "Cough"),
    ("Disease", "SoreThroat"),
])

# Define the conditional probabilitie
# Since no prior is given we will use the prior of 0.5
# Prior Prob of Disease
cp_disease = TabularCPD(
    variable = "Disease", variable_card = 2, values=[[0.5],[0.5]],
    state_names={"Disease":["Flu","Strep"]}
)

# Conditional probability of Symptops given the disease
# Conditional probability of fever given disease
cp_fever = TabularCPD(
    variable="Fever", variable_card=2,
    values=[[0.4, 0.6], # P(Fever = 0 | Disease)
            [0.6, 0.4]],# P(Fever = 1 | Disease)
    evidence=["Disease"], evidence_card=[2],
    state_names={"Fever": ["No", "Yes"], "Disease": ["Flu", "Strep"]}
)

# Conditional probability of cough given disease
cp_cough = TabularCPD(
    variable="Cough", variable_card=2,
    values=[[0.5, 0.2],
            [0.5, 0.8]],
    evidence=["Disease"], evidence_card=[2],
    state_names={"Cough": ["No", "Yes"], "Disease": ["Flu", "Strep"]}
)

# Conditional probability of sore throat given disease
cp_sore_throat = TabularCPD(
    variable="SoreThroat", variable_card=2,
    values=[[0.7, 0.4], 
            [0.3, 0.6]],
    evidence=["Disease"], evidence_card=[2],
    state_names={"SoreThroat": ["No", "Yes"], "Disease": ["Flu", "Strep"]}
)

# Add CPDs to the model
model.add_cpds(cp_disease, cp_fever, cp_cough, cp_sore_throat)

# Check if the model is valid
assert model.check_model(), "The Bayesian Network model is incorrect!"

# Perform inference
inference = VariableElimination(model)

# Query the probability of having Strep given that the patient has a sore throat
result = inference.query(variables=["Disease"], evidence={"SoreThroat": "Yes"})
# result = inference.query(variables=["Disease"], evidence={"Fever": 1, "Cough": 1, "SoreThroat": 1})
print(result)