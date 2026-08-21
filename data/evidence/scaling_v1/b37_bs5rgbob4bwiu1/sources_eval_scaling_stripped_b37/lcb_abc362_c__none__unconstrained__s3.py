import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    # input_data[1:] contains L1, R1, L2, R2, ...
    # We use slice steps to separate Ls and Rs
    L = [int(x) for x in input_data[1::2]]
    R = [int(x) for x in input_data[2::2]]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values X_i = L_i
        # We need to increase the total sum from min_sum to 0.
        # The total amount we need to add is -min_sum.
        diff = -min_sum
        
        # For each i, the maximum we can add to L_i is (R_i - L_i).
        # We use accumulate to distribute the 'diff' across the ranges.
        # For each i, we add min(diff_remaining, R_i - L_i).
        
        # Calculate the capacity of each slot: R_i - L_i
        capacities = [r - l for l, r in zip(L, R)]
        
        # Prefix sum of capacities to see how much we have "filled" so far
        pref_caps = accumulate(capacities)
        
        # The amount added to X_i is the intersection of the range 
        # [pref_caps[i-1], pref_caps[i]] and [0, diff].
        # Since we cannot use loops, we calculate the added value for each i:
        # added_i = min(R_i - L_i, max(0, diff - (pref_caps[i-1] if i>0 else 0)))
        
        # Using a list comprehension to calculate X_i for all i:
        # We use a helper function or a map-like logic to handle the index
        # But since we can't use loops, we can use the fact that:
        # X_i = L_i + (amount of diff absorbed by index i)
        # The amount absorbed by index i is:
        # min(capacity_i, max(0, diff - sum_of_capacities_before_i))
        
        # To avoid the loop/index issue with accumulate, we can use the 
        # property that the total added to index i is:
        # min(pref_caps[i], diff) - min(pref_caps[i-1], diff)
        
        # We create a list of prefix sums and append 0 at the start
        p = [0] + list(pref_caps)
        
        # X_i = L_i + min(p[i+1], diff) - min(p[i], diff)
        # We use a list comprehension to generate the sequence X
        res = [L[i] + min(p[i+1], diff) - min(p[i], diff) for i in range(N)]
        
        print("Yes")
        print(*(res))
    else:
        print("No")

if __s_main__ := True:
    solve()