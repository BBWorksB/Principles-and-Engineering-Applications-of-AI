import numpy as np

# Initialize the board
board = np.zeros((6, 7), dtype=int)

# Constants for players
PLAYER = 1
AI = 2

def print_board(board):
    for row in board:
        print(' '.join(str(cell) for cell in row))
        print("-------------")
    print()
    print("#" * 25)

def is_valid_move(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(5, -1, -1):
        if board[r][col] == 0:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Check horizontal, vertical, and diagonal locations for a win
    for c in range(7-3):
        for r in range(6):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    for c in range(7):
        for r in range(3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    for c in range(7-3):
        for r in range(3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    for c in range(7-3):
        for r in range(3, 6):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER if piece == AI else AI
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 4
    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, 3])]
    center_count = center_array.count(piece)
    score += center_count * 3
    for r in range(6):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(4):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)
    for c in range(7):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)
    for r in range(3):
        for c in range(4):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    for r in range(3):
        for c in range(4):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or len(get_valid_locations(board)) == 0

def get_valid_locations(board):
    valid_locations = []
    for col in range(7):
        if is_valid_move(board, col):
            valid_locations.append(col)
    return valid_locations

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)
    
    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, AI):
                return (None, 1000000)
            elif winning_move(board, PLAYER):
                return (None, -1000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI))
    
    if maximizingPlayer:
        value = -np.inf
        best_col = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    
    else:
        value = np.inf
        best_col = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

def pick_best_move(board, piece):
    best_col, _ = minimax(board, 5, -np.inf, np.inf, True)
    return best_col

# Main game loop
game_over = False
turn = np.random.randint(2)

while not game_over:
    print_board(board)

    if turn == PLAYER:
        col = int(input("Player 1 Make your Selection (0-6):"))
        while col < 0 or col >= 7 or not is_valid_move(board, col):
            col = int(input("Invalid selection. Make your Selection (0-6):"))
        row = get_next_open_row(board, col)
        drop_piece(board, row, col, PLAYER)
        if winning_move(board, PLAYER):
            print("PLAYER 1 WINS!")
            game_over = True
    else:
        col = pick_best_move(board, AI)
        if is_valid_move(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI)
            print(f"AI places piece in column {col}, row {row}")
            if winning_move(board, AI):
                print("AI WINS!")
                game_over = True

    turn += 1
    turn = turn % 2

    if len(get_valid_locations(board)) == 0:
        print("IT'S A TIE!")
        game_over = True
