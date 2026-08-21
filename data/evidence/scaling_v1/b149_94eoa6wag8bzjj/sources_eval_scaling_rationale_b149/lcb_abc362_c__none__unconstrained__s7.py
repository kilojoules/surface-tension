import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    pairs = [int(x) for x in input_data[1:]]
    L = pairs[0::2]
    R = pairs[1::2]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A solution exists if and only if 0 is within [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start by setting every X_i = L_i.
        # We need to increase the total sum from sum_L to 0.
        # The total amount we need to add is -sum_L.
        needed = -sum_L
        
        # For each i, the maximum we can add to L_i is (R_i - L_i).
        # We use accumulate to track how much of the 'needed' amount 
        # has been absorbed by the elements processed so far.
        capacities = [r - l for l, r in zip(L, R)]
        
        # cumulative_capacity[i] is the total room available in the first i elements.
        # We can take min(needed, cumulative_capacity[i]) as the total added so far.
        # The amount added to the i-th element specifically is:
        # min(needed, cum_cap[i]) - min(needed, cum_cap[i-1])
        cum_cap = list(accumulate(capacities))
        
        # Calculate X_i = L_i + amount_added_to_i
        # We use a list comprehension to avoid for-loops.
        # The amount added to index i is:
        # min(needed, cum_cap[i]) - (min(needed, cum_cap[i-1]) if i > 0 else 0)
        X = [
            L[i] + (min(needed, cum_cap[i]) - (min(needed, cum_cap[i-1]) if i > 0 else 0))
            for i in range(N)
        ]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()