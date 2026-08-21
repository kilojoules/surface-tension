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

    # DP state: (max_wins_if_T_played_R, max_wins_if_T_played_P, max_wins_if_T_played_S)
    # Initialize with -infinity for the first step to allow any starting move
    initial_state = (-float('inf'), -float('inf'), -float('inf'))

    def transition(state, aoki_move):
        # Current state: (r, p, s)
        # We calculate the new max wins for each possible move Takahashi makes now
        
        # If Takahashi plays Rock ('R')
        # He wins if Aoki played Scissors ('S'), draws if Aoki played Rock ('R')
        # He cannot play 'R' if he played 'R' in the previous step.
        res_r = (
            (state[1] + (1 if aoki_move == 'S' else 0)) if aoki_move in 'RS' else -float('inf'),
            (state[2] + (1 if aoki_move == 'S' else 0)) if aoki_move in 'RS' else -float('inf')
        )
        # The above logic is slightly flawed because we need to check if the move is legal.
        # Let's redefine:
        
        # For each possible move T can make: R, P, S
        # T can make move M if M != prev_M AND T does not lose to Aoki.
        # T wins if M beats Aoki.
        
        def get_score(m_t, m_a):
            if m_t == win_move[m_a]: return 1
            if m_t == draw_move[m_a]: return 0
            return -float('inf') # T lost

        # New scores for T playing R, P, S respectively
        # T plays R: must have played P or S before.
        score_r = get_score('R', aoki_move)
        next_r = max(state[1], state[2]) + score_r if score_r != -float('inf') else -float('inf')
        
        # T plays P: must have played R or S before.
        score_p = get_score('P', aoki_move)
        next_p = max(state[0], state[2]) + score_p if score_p != -float('inf') else -float('inf')
        
        # T plays S: must have played R or P before.
        score_s = get_score('S', aoki_move)
        next_s = max(state[0], state[1]) + score_s if score_s != -float('inf') else -float('inf')
        
        # Special case for the first move: since there is no "previous" move,
        # the condition "different from move i-1" is vacuously true.
        # However, the reduce starts with initial_state. 
        # To handle the first move correctly, we check if we are at the start.
        return (next_r, next_p, next_s)

    # To handle the first move correctly without a loop, we can't easily check "index == 0" 
    # inside reduce without enumerate. Let's use enumerate.
    
    def transition_with_idx(state, pair):
        idx, aoki_move = pair
        
        def get_score(m_t, m_a):
            if m_t == win_move[m_a]: return 1
            if m_t == draw_move[m_a]: return 0
            return -float('inf')

        # For the first move (idx == 0), the previous state doesn't restrict the move.
        # We can simulate this by treating the previous state as 0 for all.
        prev_r, prev_p, prev_s = state
        
        # T plays R
        s_r = get_score('R', aoki_move)
        n_r = (max(prev_p, prev_s) if idx > 0 else 0) + s_r if s_r != -float('inf') else -float('inf')
        
        # T plays P
        s_p = get_score('P', aoki_move)
        n_p = (max(prev_r, prev_s) if idx > 0 else 0) + s_p if s_p != -float('inf') else -float('inf')
        
        # T plays S
        s_s = get_score('S', aoki_move)
        n_s = (max(prev_r, prev_p) if idx > 0 else 0) + s_s if s_s != -float('inf') else -float('inf')
        
        return (n_r, n_p, n_s)

    final_state = reduce(transition_with_idx, enumerate(S), initial_state)
    print(int(max(final_state)))

if __name__ == "__main__":
    solve()