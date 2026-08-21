import sys
from itertools import accumulate

def solve():
    # Read all input at once and split into a list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element, followed by L_i, R_i pairs
    N = int(input_data[0])
    pairs = [ (int(input_data[2*i + 1]), int(input_data[2*i + 2])) for i in range(N) ]
    
    # Calculate the minimum and maximum possible sums of X_i
    # min_sum = sum(L_i), max_sum = sum(R_i)
    # We use a generator expression inside sum()
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We need to find X_i such that L_i <= X_i <= R_i and sum(X_i) = 0.
        # Let X_i = L_i + d_i, where 0 <= d_i <= (R_i - L_i).
        # Then sum(L_i + d_i) = 0  =>  sum(d_i) = -sum(L_i).
        # Let Target = -min_sum. We need to distribute Target across d_i.
        target = -min_sum
        
        # For each i, the maximum we can add to L_i is (R_i - L_i).
        # We take the minimum of the remaining target and the available headroom.
        # We use a helper function to maintain the state of the remaining target.
        def distribute(state, pair):
            rem, results = state
            L, R = pair
            headroom = R - L
            add = min(rem, headroom)
            return (rem - add, results + [L + add])
        
        # Use accumulate to iterate through the pairs while updating the remaining target.
        # The initial state is (target, [])
        final_state = list(accumulate(pairs, distribute, initial=(target, [])))[-1]
        X = final_state[1]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()