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
    # Losing move for Takahashi (Aoki wins):
    # Aoki R (0) -> Takahashi S (2)
    # Aoki P (1) -> Takahashi R (0)
    # Aoki S (2) -> Takahashi P (1)
    
    # We define a helper to get the result of Takahashi playing move 't' against Aoki's move 'a'
    # Returns (is_allowed, score)
    # is_allowed: Takahashi must not lose.
    # score: 1 if Takahashi wins, 0 if draw.
    
    # Move indices: 0: R, 1: P, 2: S
    # Aoki's move mapping
    aoki_map = {'R': 0, 'P': 1, 'S': 2}
    
    # win_map[aoki_move] = takahashi_winning_move
    win_map = {0: 1, 1: 2, 2: 0}
    # lose_map[aoki_move] = takahashi_losing_move
    lose_map = {0: 2, 1: 0, 2: 1}

    # Initial state: dp[move] = max wins ending with that move.
    # For the first move, we initialize based on whether it's a win or draw.
    # If Takahashi loses, we set it to -infinity.
    
    def get_initial_dp(first_move_aoki):
        a = aoki_map[first_move_aoki]
        return [
            1 if t == win_map[a] else (0 if t != lose_map[a] else -float('inf'))
            for t in range(3)
        ]

    def transition(dp, char_aoki):
        a = aoki_map[char_aoki]
        # For each possible move t for Takahashi in the current round:
        # He must not lose: t != lose_map[a]
        # He must have played a different move prev_t in the previous round: prev_t != t
        # Score is 1 if t == win_map[a], else 0
        
        return [
            max([dp[prev_t] + (1 if t == win_map[a] else 0) 
                 for prev_t in range(3) if prev_t != t])
            if t != lose_map[a] else -float('inf')
            for t in range(3)
        ]

    # Use reduce to iterate through the string S starting from the second character
    final_dp = reduce(transition, S[1:], get_initial_dp(S[0]))
    
    # The answer is the maximum value in the final DP state
    print(int(max(final_dp)))

if __name__ == "__main__":
    solve()