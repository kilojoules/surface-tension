import sys
from functools import reduce

def solve():
    # Read N and S from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = input_data[1]

    # Mapping of Aoki's move to the move Takahashi needs to WIN and the move to DRAW
    # Win: R->P, P->S, S->R | Draw: R->R, P->P, S->S
    # Takahashi cannot lose, so he only chooses between Winning or Drawing.
    win_move = {'R': 'P', 'P': 'S', 'S': 'R'}
    draw_move = {'R': 'R', 'P': 'P', 'S': 'S'}

    # State for reduce: (previous_move, total_wins)
    # We use a helper function to determine the best move for the current game.
    # If the winning move is different from the previous move, he takes it.
    # Otherwise, he must take the drawing move (which is guaranteed to be different 
    # from the winning move).
    
    def own_move(state, aoki_move):
        prev_move, wins = state
        w = win_move[aoki_move]
        d = draw_move[aoki_move]
        
        # If it's the first move, he always tries to win.
        if prev_move is None:
            return (w, wins + 1)
        
        # If winning move is different from previous, he wins.
        if w != prev_move:
            return (w, wins + 1)
        # Otherwise, he must draw (since he cannot lose and cannot repeat move).
        # Note: d is always different from w.
        return (d, wins)

    # Use reduce to process the string S and accumulate the score
    final_state = reduce(own_move, S, (None, 0))
    
    # The result is the second element of the final state tuple
    print(final_state[1])

if __name__ == "__main__":
    solve()