from connectfourminimaxquestions import minimax, minimax_alpha_beta 
import time 

board = [
    ['.', '.', '.', '.', '.', '.'], 
    ['x', 'x', 'o', '.', '.', '.'], 
    ['o', 'o', 'o', 'x', '.', '.'], 
    ['o', 'x', 'o', 'x', 'x', '.'], 
    ['x', 'o', 'x', 'o', 'o', 'o'], 
    ['o', 'x', 'x', 'o', 'x', 'x'], 
] # X's turn 

# Use minimax (assuming it is X's turn, so isMaximizing=True) to choose the best move. 
starttime = time.time() 
# score, best_move = minimax(board, True) 
score, best_move = minimax_alpha_beta(board, True, -float('inf'), float('inf')) # (2, 4) score 1 
endtime = time.time() 
print("Time taken:", endtime - starttime) 
print("Best move for X:", best_move, "with score:", score) 

board = [
    ['.', '.', '.', '.', '.', '.'], 
    ['.', 'x', 'o', 'x', '.', '.'], 
    ['o', 'o', 'o', 'x', 'o', '.'], 
    ['o', 'x', 'o', 'x', 'x', '.'], 
    ['x', 'o', 'x', 'o', 'o', '.'], 
    ['o', 'x', 'x', 'o', 'x', 'x'], 
] # X's turn 

# Use minimax (assuming it is X's turn, so isMaximizing=True) to choose the best move. 
starttime = time.time() 
# score, best_move = minimax(board, True) 
score, best_move = minimax_alpha_beta(board, True, -float('inf'), float('inf')) # (0, 3) score 1 
endtime = time.time() 
print("Time taken:", endtime - starttime) 
print("Best move for X:", best_move, "with score:", score) 

board = [
    ['.', '.', '.', '.', '.', '.'], 
    ['o', 'x', 'x', '', '.', '.'], 
    ['o', 'o', 'o', 'x', 'o', '.'], 
    ['o', 'o', 'x', 'x', 'x', '.'], 
    ['x', 'o', 'x', 'o', 'o', '.'], 
    ['o', 'x', 'x', 'o', 'x', 'x'], 
] # X's turn 

# Use minimax (assuming it is X's turn, so isMaximizing=True) to choose the best move. 
starttime = time.time() 
# score, best_move = minimax(board, True) 
score, best_move = minimax_alpha_beta(board, True, -float('inf'), float('inf')) # (0, 3) score 1 
endtime = time.time() 
print("Time taken:", endtime - starttime) 
print("Best move for X:", best_move, "with score:", score) 
