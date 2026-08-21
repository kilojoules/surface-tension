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
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(L)
    max_sum = sum(R)
    
    # Condition for existence: 0 must be within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with all X_i = L_i. 
    # We need to add 'diff' to the elements to make the sum 0.
    diff = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We want to distribute 'diff' across these capacities.
    capacities = [r - l for l, r in zip(L, R)]
    
    # Use accumulate to find how much of the diff is consumed by each element.
    # The amount added to X_i is min(capacity_i, remaining_diff).
    # Let S be the prefix sum of capacities.
    # The total amount added up to index i is min(diff, S_i).
    # The amount added specifically to index i is min(diff, S_i) - min(diff, S-1).
    
    S = list(accumulate(capacities))
    
    # Calculate X_i = L_i + (amount contributed to the sum)
    # amount_i = min(diff, S[i]) - (min(diff, S[i-1]) if i > 0 else 0)
    
    # To avoid if/else in list comprehension for the index 0, 
    # we can prepend 0 to S.
    S_shifted = [0] + S
    
    # Calculate the actual values of X
    # X_i = L_i + min(diff, S_shifted[i+1]) - min(diff, S_shifted[i])
    X = [
        L[i] + min(diff, S_shifted[i+1]) - min(diff, S_shifted[i])
        for i in range(N)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()