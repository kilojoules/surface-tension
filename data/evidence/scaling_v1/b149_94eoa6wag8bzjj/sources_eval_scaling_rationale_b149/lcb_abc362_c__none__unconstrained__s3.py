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
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start by setting X_i = L_i. 
    # We need to increase the total sum from min_sum to 0.
    # The total amount we need to add is -min_sum.
    needed = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to track how much of the 'needed' amount is consumed.
    # capacities = [R_i - L_i for i in range(N)]
    capacities = [r - l for l, r in zip(L, R)]
    
    # cumulative_capacities[i] is the total room for increase up to index i.
    cum_cap = list(accumulate(capacities))
    
    # For each i, the amount added to L_i is:
    # min(capacity_i, needed - sum_of_previous_capacities)
    # This can be expressed as: 
    # current_total_added = min(needed, cum_cap[i])
    # amount_for_i = current_total_added - (current_total_added_of_previous)
    
    # Calculate the total added at each step, capped at 'needed'
    totals_added = [min(needed, val) for val in cum_cap]
    
    # The amount added to X_i is totals_added[i] - totals_added[i-1]
    # We prepend 0 to totals_added to handle the i=0 case easily.
    diffs = [curr - prev for curr, prev in zip(totals_added, [0] + totals_added[:-1])]
    
    # Final X_i = L_i + diff_i
    X = [l + d for l, d in zip(L, diffs)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()