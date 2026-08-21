import sys
from itertools import accumulate

def solve():
    # Read all input data
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
    # The current sum is min_sum. We need to add 'diff' to reach 0.
    diff = 0 - min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to find how much of 'diff' is absorbed by each element.
    # capacities = [R_i - L_i for i in range(N)]
    capacities = [r - l for l, r in zip(L, R)]
    
    # cumulative_capacities[i] is the total room available in the first i+1 elements.
    cumulative_capacities = list(accumulate(capacities))
    
    # For each i, the amount added to L_i is:
    # min(capacity_i, diff - sum of previous capacities)
    # This can be calculated as: 
    # current_total_added = min(diff, cumulative_capacities[i])
    # amount_added_to_i = current_total_added - (current_total_added_of_previous)
    
    # We calculate the total amount added up to index i, capped at 'diff'.
    totals = [min(diff, c) for c in cumulative_capacities]
    
    # The actual value X_i is L_i + (totals[i] - totals[i-1])
    # We prepend 0 to totals to handle the i=0 case easily.
    totals_with_zero = [0] + totals
    X = [L[i] + (totals_with_zero[i+1] - totals_with_zero[i]) for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()