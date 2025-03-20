import copy 
import time 

# Check if a given player ('x' or 'o') has 4 in a row on the board.
def check_win(board, player):
    n = len(board)  # Board is n x n (here, n=6)
    
    # Check horizontal (rows)
    for i in range(n):
        for j in range(n - 3):
            if all(board[i][j+k] == player for k in range(4)):
                return True

    # Check vertical (columns)
    for j in range(n):
        for i in range(n - 3):
            if all(board[i+k][j] == player for k in range(4)):
                return True

    # Check diagonal down-right
    for i in range(n - 3):
        for j in range(n - 3):
            if all(board[i+k][j+k] == player for k in range(4)):
                return True

    # Check diagonal up-right
    for i in range(3, n):
        for j in range(n - 3):
            if all(board[i-k][j+k] == player for k in range(4)):
                return True

    return False

# Terminal check: game over if one player wins or no moves left.
def is_terminal(board):
    if check_win(board, 'x') or check_win(board, 'o'):
        return True
    if not any('.' in row for row in board):
        return True  # Board is full: draw.
    return False

# Evaluate terminal state: +1 if X wins, -1 if O wins, 0 otherwise.
def evaluate(board):
    if check_win(board, 'x'):
        return 1 # wins 
    elif check_win(board, 'o'):
        return -1 # loses 
    else:
        return 0 # draw 

# Get all legal moves (empty cells) as (row, col) tuples.
def get_legal_moves(board):
    moves = []
    rows = len(board)
    columns = len(board[0])
    # Loop through each column.
    for col in range(columns):
        # If the top cell is not empty, the column is full.
        if board[0][col] != '.':
            continue
        # Otherwise, find the bottom-most empty cell in this column.
        for row in range(rows - 1, -1, -1):
            if board[row][col] == '.':
                moves.append((row, col))
                break  # Stop after finding the first empty cell.
    return moves

def apply_move(board, move, player):
    new_board = copy.deepcopy(board)
    i, j = move
    new_board[i][j] = player
    return new_board

# The minimax algorithm:
# Minimax algorithm without pruning
def minimax(board, isMaximizing, depth=4):
    if is_terminal(board) or depth == 0:
        return evaluate(board), None

    if isMaximizing:
        maxEval, best_move = -float('inf'), None
        for move in get_legal_moves(board):
            eval, _ = minimax(apply_move(board, move, 'x'), False, depth - 1)
            if eval > maxEval:
                maxEval, best_move = eval, move
        return maxEval, best_move
    else:
        minEval, best_move = float('inf'), None
        for move in get_legal_moves(board):
            eval, _ = minimax(apply_move(board, move, 'o'), True, depth - 1)
            if eval < minEval:
                minEval, best_move = eval, move
        return minEval, best_move

# Minimax algorithm with pruning
def minimax_alpha_beta(board, isMaximizing, alpha, beta, depth=4):
    if is_terminal(board) or depth == 0:
        return evaluate(board), None

    if isMaximizing:
        maxEval, best_move = -float('inf'), None
        for move in get_legal_moves(board):
            eval, _ = minimax_alpha_beta(apply_move(board, move, 'x'), False, alpha, beta, depth - 1)
            if eval > maxEval:
                maxEval, best_move = eval, move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval, best_move
    else:
        minEval, best_move = float('inf'), None
        for move in get_legal_moves(board):
            eval, _ = minimax_alpha_beta(apply_move(board, move, 'o'), True, alpha, beta, depth - 1)
            if eval < minEval:
                minEval, best_move = eval, move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, best_move

if __name__ == "__main__": 

    # The current 6x6 board. 'x' and 'o' represent pieces; '.' represents an empty cell. 
    board = [
        ['.', '.', '.', '.', '.', '.'], 
        ['x', 'x', 'o', '.', '.', '.'], 
        ['o', 'o', 'o', 'x', '.', '.'], 
        ['o', 'x', 'o', 'x', 'x', '.'], 
        ['x', 'o', 'x', 'o', 'o', 'o'], 
        ['o', 'x', 'x', 'o', 'x', 'x'], 
    ] # X's turn 

    # Use minimax (assuming it is X's turn, so isMaximizing=True) to choose the best move. 
    print("Minimax without Alpha-Beta Pruning:")
    starttime = time.time() 
    score, best_move = minimax(board, True) 
    endtime = time.time()
    print("Time taken:", endtime - starttime) 
    print("Best move for X:", best_move, "with score:", score)

    print("\nMinimax with Alpha-Beta Pruning:")
    starttime = time.time()
    score, best_move = minimax_alpha_beta(board, True, -float('inf'), float('inf')) 
    endtime = time.time() 
    print("Time taken:", endtime - starttime) 
    print("Best move for X:", best_move, "with score:", score) 
