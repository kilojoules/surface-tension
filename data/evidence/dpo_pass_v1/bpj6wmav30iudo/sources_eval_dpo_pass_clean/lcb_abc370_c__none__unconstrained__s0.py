import sys
from functools import reduce

def solve():
    # Read input strings
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    # We need to change S[i] to T[i] for all i where S[i] != T[i]
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To make the array X lexicographically smallest, we want the strings 
    # at the beginning of X to be as small as possible.
    # This means we should prioritize changing characters at the 
    # leftmost positions to their target values T[i] ONLY IF T[i] < S[i].
    # However, the problem asks for the lexicographically smallest ARRAY X.
    # X_1 is the first string. To make X_1 smallest, we should change the 
    # first possible character that makes the string smaller.
    # Actually, the optimal strategy to make the sequence X lexicographically 
    # smallest is:
    # 1. Identify all indices i where S[i] != T[i].
    # 2. To make X_1 smallest, we want to change the first index i where T[i] < S[i].
    # 3. If no such i exists (all T[i] > S[i]), we must change the last index i 
    #    where T[i] > S[i] to keep the string as small as possible for as long as possible.
    # Wait, the simplest greedy approach for lexicographical smallest X:
    # At each step, we want the resulting string to be the smallest possible.
    # We have a set of indices that MUST be changed.
    # To make the current string smallest, we should change the first index i 
    # where T[i] < S[i]. If we do that, the string becomes smaller.
    # If we only have indices where T[i] > S[i], changing any of them makes 
    # the string larger, so we change the last one to delay the "increase".
    
    # Correct Greedy Strategy for Lexicographical Smallest X:
    # Let U be the set of indices where S is not yet equal to T.
    # If there is any i in U such that T[i] < S[i], the best move is to 
    # change the smallest such i.
    # If for all i in U, T[i] > S[i], the best move is to change the 
    # largest such i.
    
    # Since we cannot use loops or recursion, we use reduce to simulate the process.
    # The state is (current_S, remaining_indices, history_X)
    
    def get_next_state(state, _):
        curr_s, remaining, history = state
        if not remaining:
            return state
        
        # Find indices where T[i] < curr_s[i]
        smaller = [i for i in remaining if T[i] < curr_s[i]]
        
        # Determine which index to change
        # If 'smaller' is not empty, take the minimum index to make the string 
        # smallest at the earliest position.
        # If 'smaller' is empty, take the maximum index to keep the string 
        # smallest at the earliest positions.
        idx = min(smaller) if smaller else max(remaining)
        
        # Create the new string
        new_s = curr_s[:idx] + T[idx] + curr_s[idx+1:]
        
        # Update remaining indices
        new_remaining = [i for i in remaining if i != idx]
        
        return (new_s, new_remaining, history + [new_s])

    # Use reduce to iterate M times (where M is the number of differences)
    # We pass a dummy range to reduce to act as our loop counter
    final_state = reduce(get_next_state, range(len(diff_indices)), (S, diff_indices, []))
    
    # Output results
    X = final_state[2]
    sys.stdout.write(f"{len(X)}\n" + "\n".join(X) + "\n" if X else "0")

if __name__ == "__main__":
    solve()