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
    # Win: R->P, P->S, S->R
    # Draw: R->R, P->P, S->S
    # Note: Takahashi cannot lose, so he only chooses between Win and Draw.
    winning_move = {'R': 'P', 'P': 'S', 'S': 'R'}
    drawing_move = {'R': 'R', 'P': 'P', 'S': 'S'}

    # DP state: (score_if_last_was_R, score_if_last_was_P, score_if_last_was_S)
    # We use a tuple to store the maximum wins ending in R, P, and S respectively.
    # Initialize with 0.
    initial_state = (0, 0, 0)
    
    # Move mapping for indexing the state tuple
    move_idx = {'R': 0, 'P': 1, 'S': 2}

    def transition(state, aoki_move):
        # current_state = (r_score, p_score, s_score)
        # We calculate the new score for each possible move Takahashi makes.
        # Takahashi's move T must be different from the previous move.
        # T must be either the winning_move or the drawing_move for aoki_move.
        
        win_T = winning_move[aoki_move]
        draw_T = drawing_move[aoki_move]
        
        # Calculate new scores for R, P, and S
        # If Takahashi plays 'R':
        # He wins if aoki_move == 'S', draws if aoki_move == 'R'.
        # He can only play 'R' if the previous move was 'P' or 'S'.
        
        def get_score(t_move):
            # Score gained this round
            gain = 1 if t_move == winning_move[aoki_move] else 0
            # Check if t_move is a valid move (cannot lose)
            if t_move != winning_move[aoki_move] and t_move != drawing_move[aoki_move]:
                return -float('inf')
            
            # Max score from previous states where move was different
            # state is (score_R, score_P, score_S)
            prev_scores = [state[i] for i in range(3) if i != move_idx[t_move]]
            return max(prev_scores) + gain

        return (
            get_score('R'),
            get_score('P'),
            get_score('S')
        )

    # Use reduce to simulate the DP process across the string S
    final_state = reduce(transition, S, initial_state)
    
    # The answer is the maximum value in the final state tuple
    print(max(final_state))

if __name__ == "__main__":
    solve()