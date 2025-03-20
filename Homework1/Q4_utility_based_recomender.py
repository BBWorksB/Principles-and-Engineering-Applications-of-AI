# Step 1: Define the function `recommend_products`
def recommend_products(user_data, products, n):
    """
    Recommend top n products based on user preferences and past interactions.
    
    Args:
        user_data (dict): User's past ratings and preferences.
        products (list): List of product dictionaries.
        n (int): Number of recommendations to return.
    
    Returns:
        list: Top n recommended products with utility scores.
    """
    # Step 2: Extract user preferences and past ratings
    user_ratings = user_data.get("ratings", {})
    user_preferences = user_data.get("preferences", {})
    
    # Step 3: Initialize an empty list to store utility scores
    utility_scores = []

    # Step 4: Iterate through each product in the products list
    for product in products:
        # Step 5: Calculate the utility score for the current product
        utility_score = calculate_utility(user_ratings, user_preferences, product)
        
        # Step 6: Append the product and its utility score to the utility_scores list
        utility_scores.append((product, utility_score))

    # Step 7: Sort the utility_scores list by score in descending order
    utility_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 8: Return the top n products
    return utility_scores[:n]

# Step 9: Define a helper function `calculate_utility`
def calculate_utility(user_ratings, user_preferences, product):
    """
    Calculate utility score for a product.
    
    Args:
        user_ratings (dict): User's past ratings.
        user_preferences (dict): User's preferences.
        product (dict): Product details.
    
    Returns:
        float: Utility score for the product.
    """
    # Step 10: Implement the scoring logic
    category_score = 1.0 if product["category"] in user_preferences.get("category", []) else 0.5
    
    # If the user has rated this product before, use it; otherwise, take product rating
    rating_score = user_ratings.get(product["id"], product["rating"]) / 5.0  # Normalize rating (0-1)
    
    # Apply a price penalty: Cheaper products are slightly preferred
    price_penalty = 1.0 / (1.0 + (product["price"] / 100.0))  # Normalized price factor
    
    # Combine the scores into a final utility score
    utility = (category_score * 0.4) + (rating_score * 0.5) + (price_penalty * 0.1)
    return utility

# Step 11: Test the function
user_data = {
    "ratings": {"product1": 5, "product2": 3},
    "preferences": {"category": ["electronics", "books"]}
}

products = [
    {"id": "product1", "category": "electronics", "price": 499, "rating": 4.5},
    {"id": "product2", "category": "books", "price": 15, "rating": 4.0},
    {"id": "product3", "category": "clothing", "price": 50, "rating": 3.8},
    {"id": "product4", "category": "electronics", "price": 1000, "rating": 4.7},
    {"id": "product5", "category": "books", "price": 25, "rating": 4.2}
]

# Get top 3 recommendations
recommendations = recommend_products(user_data, products, 3)
for product, score in recommendations:
    print(f"Recommended: {product['id']} | Utility Score: {score:.2f}")
