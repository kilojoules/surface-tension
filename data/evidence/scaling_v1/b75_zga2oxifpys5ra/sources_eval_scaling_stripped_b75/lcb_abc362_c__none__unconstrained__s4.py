import sys
from itertools import accumulate

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of (L, R)
    pairs = [
        (int(input_data[2*i + 1]), int(input_data[2*i + 2])) 
        for i in range(N)
    ]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We need to distribute the remaining sum needed to reach 0
        # Starting from the minimum possible sum, we need to add 'target'
        target = -min_sum
        
        # For each i, the maximum we can add to L_i is (R_i - L_i)
        # We use accumulate to greedily fill the target
        # current_sum tracks the total amount added so far
        def get_added_value(acc, pair):
            L, R = pair
            capacity = R - L
            added = min(capacity, target - acc)
            return acc + added

        # Calculate the total added amount at each step
        # accumulate returns an iterator of the running totals
        totals = list(accumulate(pairs, get_added_value, initial=0))
        
        # The amount added to X_i is (totals[i+1] - totals[i])
        # X_i = L_i + (totals[i+1] - totals[i])
        X = [
            pairs[i][0] + (totals[i+1] - totals[i]) 
            for i in range(N)
        ]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()