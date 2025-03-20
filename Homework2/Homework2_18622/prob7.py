import heapq
import copy
import time

class Node:
    def __init__(self, data, parent, level, fval):
        self.data = data
        self.parent = parent
        self.level = level
        self.fval = fval

    def __lt__(self, other):
        return self.fval < other.fval
    
    def state_string(self):
        """Convert state to string for hashing"""
        return ''.join([''.join(row) for row in self.data])

class AdversarialNPuzzle:
    def __init__(self, size):
        self.n = size
        self.visited = set()
        self.max_moves = 50  # Add a maximum move limit to prevent infinite games
        self.current_moves = 0
    
    def take_input(self):
        p = []
        for i in range(self.n):
            temp = input().split()
            p.append(temp)
        return p
    
    def find_blank_tile(self, state):
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] == '_':
                    return i, j
        return -1, -1  # Should not happen if the puzzle is valid
    
    def move_tile(self, state, x1, y1, x2, y2):
        new_state = copy.deepcopy(state)
        new_state[x1][y1], new_state[x2][y2] = new_state[x2][y2], new_state[x1][y1]
        return new_state
    
    def generate_child(self, node):
        children = []
        x, y = self.find_blank_tile(node.data)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in moves:
            newx, newy = x + dx, y + dy
            if 0 <= newx < self.n and 0 <= newy < self.n:
                new_state = self.move_tile(node.data, x, y, newx, newy)
                child = Node(new_state, node, node.level + 1, 0)
                children.append(child)
        return children
    
    def manhattan_distance(self, state, goal):
        distance = 0
        goal_dict = {}
        
        # Dictionary mapping each value to its position in the goal state
        for i in range(self.n):
            for j in range(self.n):
                if goal[i][j] != '_':
                    goal_dict[goal[i][j]] = (i, j)
        
        # Manhattan distance for each tile
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] != '_' and state[i][j] in goal_dict:
                    goal_x, goal_y = goal_dict[state[i][j]]
                    distance += abs(i - goal_x) + abs(j - goal_y)
        
        return distance
    
    def f_score(self, state, goal):
        return self.manhattan_distance(state, goal)
    
    def is_end(self, state, goal, opponent_goal):
        # current state matches either goal state or if max moves reached
        return state == goal or state == opponent_goal or self.current_moves >= self.max_moves
    
    def minimax(self, node, depth, is_maximizing, alpha, beta, goal, opponent_goal, memo=None):
        if memo is None:
            memo = {}
        
        # unique key for the current state and player
        state_str = ''.join([''.join(row) for row in node.data])
        key = (state_str, is_maximizing, depth)
        
        # If we've seen this state before at the same depth and for the same player, return the cached result
        if key in memo:
            return memo[key]
        
        # Terminal condition: reached max depth or game is over
        if depth == 0 or self.is_end(node.data, goal, opponent_goal):
            # For the agent (minimizing player), we want to minimize the distance to our goal
            # For the opponent (maximizing player), they want to maximize the distance to agent's goal
            if is_maximizing:
                result = -self.f_score(node.data, opponent_goal)  # Opponent tries to reach opponent_goal
            else:
                result = self.f_score(node.data, goal)  # Agent tries to minimize distance to agent's goal
            memo[key] = result
            return result
        
        children = self.generate_child(node)
        
        # No valid moves
        if not children:
            memo[key] = 0
            return 0
            
        if is_maximizing:
            max_eval = float('-inf')
            for child in children:
                eval = self.minimax(child, depth - 1, False, alpha, beta, goal, opponent_goal, memo)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            memo[key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in children:
                eval = self.minimax(child, depth - 1, True, alpha, beta, goal, opponent_goal, memo)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            memo[key] = min_eval
            return min_eval
    
    def get_best_move(self, current, is_agent_turn, goal, opponent_goal, search_depth=3):
        """Find the best move for the current player"""
        # dictionary to cache results
        memo = {}
        
        best_move = None
        
        if is_agent_turn:  # Agent (minimizing player)
            best_value = float('inf')
            for child in self.generate_child(current):
                child_state_str = ''.join([''.join(row) for row in child.data])
                
                # Skip if this state has been visited before (to avoid cycles)
                if child_state_str in self.visited:
                    continue
                    
                move_value = self.minimax(child, search_depth, False, float('-inf'), float('inf'), 
                                         goal, opponent_goal, memo)
                if move_value < best_value:
                    best_value = move_value
                    best_move = child
        else:  # Opponent (maximizing player)
            best_value = float('-inf')
            for child in self.generate_child(current):
                child_state_str = ''.join([''.join(row) for row in child.data])
                
                # Skip if this state has been visited before (to avoid cycles)
                if child_state_str in self.visited:
                    continue
                    
                move_value = self.minimax(child, search_depth, True, float('-inf'), float('inf'), 
                                         goal, opponent_goal, memo)
                if move_value > best_value:
                    best_value = move_value
                    best_move = child
        
        # If no valid moves found (all lead to visited states), pick the first available move
        if best_move is None and self.generate_child(current):
            best_move = self.generate_child(current)[0]
            
        return best_move
    
    def play(self):
        print("Enter the start state:")
        start = self.take_input()
        print("Enter the agent's goal state:")
        goal = self.take_input()
        print("Enter the opponent's goal state:")
        opponent_goal = self.take_input()
        
        # Initialize game state
        current = Node(start, None, 0, 0)
        is_agent_turn = True
        self.visited = set()  # Reset visited states
        self.current_moves = 0
        
        print("\nStarting Game:")
        for row in current.data:
            print(" ".join(row))
        print("\n")
        
        # Main game loop
        while not self.is_end(current.data, goal, opponent_goal):
            # Keep track of visited states to avoid cycles
            current_state_str = ''.join([''.join(row) for row in current.data])
            self.visited.add(current_state_str)
            
            # Make move based on player
            if is_agent_turn:
                print("Agent's move:")
                start_time = time.time()
                best_move = self.get_best_move(current, True, goal, opponent_goal)
                end_time = time.time()
                print(f"Move calculated in {end_time - start_time:.2f} seconds")
            else:
                print("Opponent's move:")
                start_time = time.time()
                best_move = self.get_best_move(current, False, goal, opponent_goal)
                end_time = time.time()
                print(f"Move calculated in {end_time - start_time:.2f} seconds")
            
            # Check if a valid move was found
            if best_move is None:
                print("No valid moves available. Game ends in a draw.")
                break
            
            # Update game state
            current = best_move
            self.current_moves += 1
            
            # Display current state
            for row in current.data:
                print(" ".join(row))
            print(f"Move {self.current_moves}/{self.max_moves}\n")
            
            # Switch player turn
            is_agent_turn = not is_agent_turn
        
        # Determine winner
        if current.data == goal:
            print("Agent wins!")
        elif current.data == opponent_goal:
            print("Opponent wins!")
        elif self.current_moves >= self.max_moves:
            print("Game ended due to maximum moves reached. It's a draw!")
        else:
            print("Game ended in a draw.")

# Create & run the game
puzzle = AdversarialNPuzzle(3)
puzzle.play()