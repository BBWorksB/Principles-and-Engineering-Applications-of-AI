# !pip install pomegranate
# !pip install pomegranate==0.15.0
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, Node, BayesianNetwork


# Define the possible states for each variable
# Season: (Winter, Spring, Summer, Fall)
# Price: (Low, Medium, High)
# Advertising: (Low, Medium, High)
# Sales: (Low, Medium, High)

# Define the prior probability distribution for Season
season = DiscreteDistribution({
    'Winter': 0.25, 'Spring': 0.25, 'Summer': 0.25, 'Fall': 0.25  # Assuming equal probability for all seasons
})

# Define the conditional probability distribution for Price given Season
price = ConditionalProbabilityTable([
    ['Winter', 'Low', 0.4], ['Winter', 'Medium', 0.4], ['Winter', 'High', 0.2],
    ['Spring', 'Low', 0.3], ['Spring', 'Medium', 0.5], ['Spring', 'High', 0.2],
    ['Summer', 'Low', 0.2], ['Summer', 'Medium', 0.5], ['Summer', 'High', 0.3],
    ['Fall', 'Low', 0.3], ['Fall', 'Medium', 0.4], ['Fall', 'High', 0.3]
], [season])  # Dependency on Season

# Define the conditional probability distribution for Advertising given Season
advertising = ConditionalProbabilityTable([
    ['Winter', 'Low', 0.5], ['Winter', 'Medium', 0.3], ['Winter', 'High', 0.2],
    ['Spring', 'Low', 0.3], ['Spring', 'Medium', 0.4], ['Spring', 'High', 0.3],
    ['Summer', 'Low', 0.2], ['Summer', 'Medium', 0.3], ['Summer', 'High', 0.5],
    ['Fall', 'Low', 0.3], ['Fall', 'Medium', 0.4], ['Fall', 'High', 0.3]
], [season])  # Dependency on Season

# Define the conditional probability table for Sales given Price and Advertising
sales = ConditionalProbabilityTable([
    ['Low', 'Low', 'Low', 0.7], ['Low', 'Low', 'Medium', 0.6], ['Low', 'Low', 'High', 0.5],
    ['Low', 'Medium', 'Low', 0.6], ['Low', 'Medium', 'Medium', 0.5], ['Low', 'Medium', 'High', 0.4],
    ['Low', 'High', 'Low', 0.5], ['Low', 'High', 'Medium', 0.4], ['Low', 'High', 'High', 0.3],
    ['Medium', 'Low', 'Low', 0.6], ['Medium', 'Low', 'Medium', 0.5], ['Medium', 'Low', 'High', 0.4],
    ['Medium', 'Medium', 'Low', 0.5], ['Medium', 'Medium', 'Medium', 0.4], ['Medium', 'Medium', 'High', 0.3],
    ['Medium', 'High', 'Low', 0.4], ['Medium', 'High', 'Medium', 0.3], ['Medium', 'High', 'High', 0.2],
    ['High', 'Low', 'Low', 0.3], ['High', 'Low', 'Medium', 0.2], ['High', 'Low', 'High', 0.1],
    ['High', 'Medium', 'Low', 0.2], ['High', 'Medium', 'Medium', 0.1], ['High', 'Medium', 'High', 0.05],
    ['High', 'High', 'Low', 0.1], ['High', 'High', 'Medium', 0.05], ['High', 'High', 'High', 0.01]
], [price, advertising])  # Dependency on Price and Advertising

# Create the Bayesian Network Nodes
s_node = Node(season, name='Season')
p_node = Node(price, name='Price')
a_node = Node(advertising, name='Advertising')
sales_node = Node(sales, name='Sales')

# Build the Bayesian Network
model = BayesianNetwork("Sales Prediction")
model.add_states(s_node, p_node, a_node, sales_node)

# Add edges to represent dependencies
model.add_edge(s_node, p_node)  # Season influences Price
model.add_edge(s_node, a_node)  # Season influences Advertising
model.add_edge(p_node, sales_node)  # Price influences Sales
model.add_edge(a_node, sales_node)  # Advertising influences Sales

# Finalize the model
model.bake()

# Perform inference (predict sales given known conditions)
observations = {'Season': 'Winter', 'Price': 'High', 'Advertising': 'High'}
predicted_sales = model.predict([observations])

# Predicted result
print("Predicted Sales Level:", predicted_sales[0][-1])

