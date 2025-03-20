import random

class NegotiationGame:
    def __init__(self, max_rounds=5):
        """ Initializes the game with random starting prices for buyer and seller. """
        self.max_rounds = max_rounds
        self.current_round = 0
        self.buyer_offer = random.randint(20, 80)
        self.seller_price = random.randint(50, 100)
        self.stochastic_threshold = 0.3

    def get_valid_moves(self, player):
        """ Returns valid moves for the given player. """
        if player == "buyer":
            return ["accept", "counter", "walk away"]
        return ["accept", "counter"]

    def evaluate_offer(self, buyer_offer, seller_price):
        """ Returns the absolute difference between offers as the cost. """
        return abs(seller_price - buyer_offer)

    def play_negotiation(self):
        """ Runs the negotiation process. AI makes decisions using Expectiminimax. """
        negotiation_history = []  

        while self.current_round < self.max_rounds:
            print(f"\nRound {self.current_round + 1}: Buyer Offer = ${self.buyer_offer}, Seller Price = ${self.seller_price}")

            if self.buyer_offer >= self.seller_price:
                print(f"Deal reached instantly at ${self.buyer_offer}")
                return

            _, buyer_move = self.expectiminimax("buyer", self.buyer_offer, self.seller_price, depth=3)
            print(f"Buyer chooses: {buyer_move}")

            if buyer_move == "accept":
                print(f"Deal reached at ${self.seller_price}")
                return
            elif buyer_move == "walk away":
                print("Buyer walked away. No deal!")
                return
            elif buyer_move == "counter":
                self.buyer_offer += random.randint(1, 5)

            _, seller_move = self.expectiminimax("seller", self.buyer_offer, self.seller_price, depth=3)
            print(f"Seller chooses: {seller_move}")

            if seller_move == "accept":
                print(f"Deal reached at ${self.buyer_offer}")
                return
            elif seller_move == "counter":
                if random.random() < self.stochastic_threshold:
                    print("Seller made an unpredictable move!")
                    self.seller_price += random.randint(5, 10)
                else:
                    self.seller_price -= random.randint(1, 5)

            negotiation_history.append((self.current_round, self.buyer_offer, self.seller_price))
            self.current_round += 1

        print("Negotiation failed!")
        print("Negotiation history:", negotiation_history)

    def expectiminimax(self, player, buyer_offer, seller_price, depth):
        if depth == 0 or buyer_offer >= seller_price:
            return self.evaluate_offer(buyer_offer, seller_price), "accept"

        if player == "buyer":
            best_value = float("inf")
            best_move = "walk away"
            for move in self.get_valid_moves(player):
                if move == "accept":
                    value = self.evaluate_offer(buyer_offer, seller_price)
                elif move == "counter":
                    value, _ = self.expectiminimax("seller", buyer_offer + 3, seller_price, depth - 1)
                else:
                    value = float("inf")

                if value < best_value:
                    best_value, best_move = value, move

            return best_value, best_move
        
        else:  # Seller's turn
            if random.random() < self.stochastic_threshold:
                return self.evaluate_offer(buyer_offer, seller_price + random.randint(5, 10)), "counter"
            
            best_value = float("-inf")
            best_move = "accept"
            for move in self.get_valid_moves(player):
                if move == "accept":
                    value = self.evaluate_offer(buyer_offer, seller_price)
                elif move == "counter":
                    value, _ = self.expectiminimax("buyer", buyer_offer, seller_price - 3, depth - 1)
                
                if value > best_value:
                    best_value, best_move = value, move

            return best_value, best_move

# Run the game
game = NegotiationGame()
game.play_negotiation()
