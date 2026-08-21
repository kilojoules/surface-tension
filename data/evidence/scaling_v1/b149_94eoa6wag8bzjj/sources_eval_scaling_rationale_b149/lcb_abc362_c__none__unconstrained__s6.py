import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    pairs = [int(x) for x in input_data[1:]]
    ls = pairs[0::2]
    rs = pairs[1::2]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(ls)
    max_sum = sum(rs)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with all X_i = L_i. 
    # We need to increase the total sum from min_sum to 0.
    # The amount we need to add is -min_sum.
    needed = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to find how much of the 'needed' sum is consumed.
    # capacities = [R_i - L_i for i in range(n)]
    capacities = [r - l for l, r in zip(ls, rs)]
    
    # cumulative_capacities[i] is the total room for increase up to index i
    cum_cap = list(accumulate(capacities))
    
    # For each i, the amount added is:
    # min(capacity_of_i, remaining_needed_after_i-1)
    # This can be expressed as: 
    # current_total_added = min(needed, cum_cap[i])
    # amount_added_at_i = current_total_added - previous_total_added
    
    # We prepend 0 to cum_cap to handle the i=0 case easily
    # But since we can't use loops, we map a function over the indices.
    # Let's calculate the total added at each step:
    totals_added = [min(needed, c) for c in cum_cap]
    
    # The actual X_i is L_i + (totals_added[i] - totals_added[i-1])
    # We handle i=0 by treating totals_added[-1] as 0.
    res = [
        ls[i] + (totals_added[i] - (totals_added[i-1] if i > 0 else 0))
        for i in range(n)
    ]
    
    print("Yes")
    print(*(res))

if __name__ == "__main__":
    solve()