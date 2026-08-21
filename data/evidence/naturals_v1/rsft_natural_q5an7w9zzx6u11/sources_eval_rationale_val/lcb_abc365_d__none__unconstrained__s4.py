import sys
from functools import reduce

def solve():
    # Read N and S from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = input_data[1]

    # Mapping of moves to indices: R=0, P=1, S=2
    # Winning relationships: Win[aoki_move] = takahashi_move_that_wins
    # Losing relationships: Lose[aoki_move] = takahashi_move_that_loses
    # Aoki: R(0), P(1), S(2)
    # Takahashi wins if: Aoki R -> T P(1), Aoki P -> T S(2), Aoki S -> T R(0)
    # Takahashi loses if: Aoki R -> T S(2), Aoki P -> T R(0), Aoki S -> T P(1)
    
    move_map = {'R': 0, 'P': 1, 'S': 2}
    aoki_moves = list(map(lambda x: move_map[x], S))
    
    # win_move[aoki] returns the move Takahashi needs to win
    win_move = {0: 1, 1: 2, 2: 0}
    # lose_move[aoki] returns the move Takahashi cannot play (because he never loses)
    lose_move = {0: 2, 1: 0, 2: 1}

    # DP state: (max_wins_if_T_played_R, max_wins_if_T_played_P, max_wins_if_T_played_S)
    # Initial state for the first move
    def init_state(first_aoki):
        # For the first move, T can play any move except the one that loses to Aoki
        # If T plays the winning move, win is 1, otherwise 0.
        # We use -float('inf') for illegal moves.
        res = [-float('inf')] * 3
        for t_move in range(3):
            if t_move != lose_move[first_aoki]:
                res[t_move] = 1 if t_move == win_move[first_aoki] else 0
        return tuple(res)

    # Transition function for reduce
    def transition(state, aoki):
        # next_state[t_move] = max(state[prev_t_move]) + (1 if t_move wins else 0)
        # condition: t_move != prev_t_move AND t_move != lose_move[aoki]
        
        # Calculate potential wins for each possible current move T
        # T cannot play lose_move[aoki]
        return tuple(
            (
                max(state[i] for i in range(3) if i != t_move) + (1 if t_move == win_move[aoki] else 0)
                if t_move != lose_move[aoki] else -float('inf')
            )
            for t_move in range(3)
        )

    # Process the sequence
    # We start the reduce from the second element onwards
    final_state = reduce(transition, aoki_moves[1:], init_state(aoki_moves[0]))
    
    # The answer is the maximum value in the final DP state
    print(int(max(final_state)))

if __name__ == "__main__":
    solve()