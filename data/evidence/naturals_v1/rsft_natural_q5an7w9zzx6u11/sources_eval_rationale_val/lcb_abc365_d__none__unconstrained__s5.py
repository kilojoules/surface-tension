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
    # Non-losing moves for Takahashi:
    # Aoki R (0) -> Takahashi R (0) or P (1)
    # Aoki P (1) -> Takahashi P (1) or S (2)
    # Aoki S (2) -> Takahashi S (2) or R (0)
    
    move_map = {'R': 0, 'P': 1, 'S': 2}
    aoki_moves = list(map(lambda x: move_map[x], S))

    # dp state: (max_wins_if_T_played_R, max_wins_if_T_played_P, max_wins_if_T_played_S)
    # Initial state: -infinity for impossible, 0 for possible.
    # For the first move, we initialize based on whether Takahashi wins or draws.
    def init_dp(first_move):
        # win_move is the move that beats first_move
        # draw_move is the move that ties first_move
        win_move = (first_move + 1) % 3
        draw_move = first_move
        
        res = [-float('inf')] * 3
        res[win_move] = 1
        res[draw_move] = 0
        return tuple(res)

    def transition(dp, aoki_move):
        # win_move beats aoki_move, draw_move ties it
        win_move = (aoki_move + 1) % 3
        draw_move = aoki_move
        
        # For each possible move T can make now (m), 
        # he must have made a different move (prev_m) previously.
        # He can only make move m if it doesn't lose to aoki_move.
        
        def get_max_wins(m):
            # T cannot play move m if it loses to aoki_move
            # Losing condition: (m + 1) % 3 == aoki_move
            if (m + 1) % 3 == aoki_move:
                return -float('inf')
            
            # T wins if m == (aoki_move + 1) % 3
            score = 1 if m == win_move else 0
            
            # T must have played a different move previously
            # prev_m != m
            prev_scores = [dp[i] for i in range(3) if i != m]
            return score + max(prev_scores)

        return tuple(map(get_max_wins, range(3)))

    # Process the first move separately to establish the base DP state
    first_dp = init_dp(aoki_moves[0])
    
    # Use reduce to iterate through the remaining moves
    final_dp = reduce(transition, aoki_moves[1:], first_dp)
    
    # The answer is the maximum value in the final DP state
    print(max(final_dp))

if __name__ == "__main__":
    solve()