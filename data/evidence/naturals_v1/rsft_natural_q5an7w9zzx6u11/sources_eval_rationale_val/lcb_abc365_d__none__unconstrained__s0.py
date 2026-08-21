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
    # Winning move for Aoki's move:
    # Aoki R (0) -> Takahashi P (1)
    # Aoki P (1) -> Takahashi S (2)
    # Aoki S (2) -> Takahashi R (0)
    # The winning move for Takahashi is (aoki_move + 1) % 3
    # The drawing move for Takahashi is aoki_move
    # Losing move is (aoki_move - 1) % 3 (which is forbidden)
    
    move_map = {'R': 0, 'P': 1, 'S': 2}
    aoki_moves = list(map(lambda x: move_map[x], S))

    # DP state: (max_wins_if_T_played_0, max_wins_if_T_played_1, max_wins_if_T_played_2)
    # Initial state: For the first move, Takahashi can play any move that doesn't lose.
    # If Aoki plays 'a', Takahashi can play 'a' (draw) or '(a+1)%3' (win).
    def init_state(first_move):
        # Takahashi cannot play (first_move - 1) % 3
        # We use -float('inf') for impossible states
        res = [-float('inf')] * 3
        res[first_move] = 0 # Draw
        res[(first_move + 1) % 3] = 1 # Win
        return tuple(res)

    def transition(state, aoki_move):
        # For each possible move T can make now (curr_t)
        # T cannot play (aoki_move - 1) % 3
        # T cannot play the same move as the previous game (prev_t)
        
        # Calculate wins for current move
        # Win if curr_t == (aoki_move + 1) % 3, else 0 (since loss is forbidden)
        
        # We calculate the new state for curr_t in [0, 1, 2]
        return tuple(
            max(
                (state[prev_t] + (1 if curr_t == (aoki_move + 1) % 3 else 0))
                for prev_t in range(3)
                if prev_t != curr_t # Condition: move i != move i+1
                and curr_t != (aoki_move - 1) % 3 # Condition: cannot lose
            ) if curr_t != (aoki_move - 1) % 3 else -float('inf')
            for curr_t in range(3)
        )

    # Process the first move separately to initialize, then reduce the rest
    first_move = aoki_moves[0]
    initial = init_state(first_move)
    
    final_state = reduce(transition, aoki_moves[1:], initial)
    
    print(int(max(final_state)))

if __name__ == "__main__":
    solve()