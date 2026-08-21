import sys
from functools import reduce

def solve():
    # Read N and S
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]

    # Mapping for moves to indices: R=0, P=1, S=2
    # Winning move for Aoki's move:
    # Aoki R (0) -> Takahashi P (1)
    # Aoki P (1) -> Takahashi S (2)
    # Aoki S (2) -> Takahashi R (0)
    # Not losing move for Aoki's move:
    # Aoki R (0) -> Takahashi R (0) or P (1)
    # Aoki P (1) -> Takahashi P (1) or S (2)
    # Aoki S (2) -> Takahashi S (2) or R (0)
    
    move_map = {'R': 0, 'P': 1, 'S': 2}
    aoki_moves = list(map(lambda x: move_map[x], S))

    # dp state: (max_wins_if_T_played_R, max_wins_if_T_played_P, max_wins_if_T_played_S)
    # Initial state: -infinity for impossible, 0 for possible.
    # For the first move, we initialize based on whether T wins or draws.
    def init_dp(first_move):
        # T wins if move is (first_move + 1) % 3
        # T draws if move is first_move
        # T loses if move is (first_move - 1) % 3
        # We only allow win or draw.
        res = [-float('inf')] * 3
        # Win
        win_move = (first_move + 1) % 3
        res[win_move] = 1
        # Draw
        draw_move = first_move
        res[draw_move] = 0
        return tuple(res)

    # The first move is handled separately to start the reduce
    first_aoki = aoki_moves[0]
    initial_state = init_dp(first_aoki)

    def transition(dp, aoki_move):
        # dp is (r, p, s) from previous step
        # current_move is the move Takahashi makes now
        # He cannot make the same move as the previous step.
        
        # For each possible current move m in {0, 1, 2}:
        # He can only make move m if he didn't make move m in the previous step.
        # He cannot lose to aoki_move.
        
        # Win condition: m == (aoki_move + 1) % 3
        # Draw condition: m == aoki_move
        # Lose condition: m == (aoki_move - 1) % 3
        
        def get_max_for_move(m):
            # Check if move m is a losing move
            if m == (aoki_move - 1) % 3:
                return -float('inf')
            
            # Calculate win value
            win_val = 1 if m == (aoki_move + 1) % 3 else 0
            
            # He must have played a different move previously
            # prev_moves are indices other than m
            prev_indices = [i for i in range(3) if i != m]
            best_prev = max(dp[prev_indices[0]], dp[prev_indices[1]])
            
            return best_prev + win_val

        return (get_max_for_move(0), get_max_for_move(1), get_max_for_move(2))

    # Process the rest of the string
    final_dp = reduce(transition, aoki_moves[1:], initial_state)
    
    print(max(final_dp))

if __name__ == "__main__":
    solve()