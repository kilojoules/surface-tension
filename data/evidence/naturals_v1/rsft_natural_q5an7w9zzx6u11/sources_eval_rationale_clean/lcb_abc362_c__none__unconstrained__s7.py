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
    if not (sum_L <= 0 <= sum_R):
        print("No")
        return

    # We start by setting X_i = L_i. 
    # We need to increase the total sum from sum_L to 0.
    # The total amount we need to add is -sum_L.
    needed = -sum_L
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to find how much of the 'needed' sum is absorbed by each element.
    # capacities = [R_i - L_i for i in range(N)]
    capacities = [r - l for l, r in zip(L, R)]
    
    # cumulative_capacities[i] is the total capacity of the first i+1 elements.
    cum_cap = list(accumulate(capacities))
    
    # For each i, the amount added to L_i is:
    # min(capacity_i, remaining_needed)
    # This can be calculated as:
    # amount_added_i = min(R_i - L_i, needed - sum_of_previous_capacities)
    # More simply: the total added to the first i elements is min(needed, cum_cap[i-1])
    # The amount added to element i is min(needed, cum_cap[i]) - min(needed, cum_cap[i-1])
    
    # We create a helper list starting with 0 to handle the i=0 case in the comprehension
    cum_cap_with_zero = [0] + cum_cap
    
    # X_i = L_i + (amount absorbed by index i)
    # amount absorbed = min(needed, cum_cap[i]) - min(needed, cum_cap[i-1])
    X = [
        L[i] + (min(needed, cum_cap_with_zero[i+1]) - min(needed, cum_cap_with_zero[i]))
        for i in range(N)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()