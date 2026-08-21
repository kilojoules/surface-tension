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
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to add 'diff' to the elements to reach a total sum of 0.
    diff = 0 - min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to find how much of 'diff' is consumed as we go.
    # capacities = [R_i - L_i for i in range(N)]
    capacities = [r - l for l, r in zip(L, R)]
    
    # cumulative_capacities[i] is the total capacity of the first i+1 elements.
    cum_cap = list(accumulate(capacities))
    
    # For each element i, the amount we add to L_i is:
    # min(capacity_i, diff - sum_of_previous_capacities)
    # More simply: the amount added to X_i is the increase in the 
    # capped cumulative sum: min(diff, cum_cap[i]) - min(diff, cum_cap[i-1])
    
    # We prepend 0 to cum_cap to handle the i=0 case easily.
    capped_cum_cap = [min(diff, val) for val in [0] + cum_cap]
    
    # X_i = L_i + (capped_cum_cap[i+1] - capped_cum_cap[i])
    X = [L[i] + (capped_cum_cap[i+1] - capped_cum_cap[i]) for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()