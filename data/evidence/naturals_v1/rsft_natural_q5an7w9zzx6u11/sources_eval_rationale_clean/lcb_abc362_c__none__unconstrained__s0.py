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
    # We use a greedy approach: fill each X_i as much as possible until 'needed' is 0.
    # To avoid loops, we calculate the prefix sums of the available capacities.
    capacities = [r - l for l, r in zip(L, R)]
    pref_cap = list(accumulate(capacities))
    
    # For each i, the amount added is:
    # min(capacity_i, remaining_needed_at_that_step)
    # The remaining needed at step i is: needed - (sum of capacities of 1 to i-1)
    # However, it's simpler to say:
    # The total added to X_i is min(capacity_i, max(0, needed - pref_cap[i-1]))
    # We handle the i=0 case by treating pref_cap[-1] as 0.
    
    # We create a helper list for pref_cap that starts with 0
    p = [0] + pref_cap[:-1]
    
    # X_i = L_i + amount_added
    # amount_added = clamp(needed - p[i], 0, capacity_i)
    # Since we know sum_L <= 0 <= sum_R, the total added will exactly be 'needed'.
    X = [
        l + max(0, min(r - l, needed - p_val))
        for l, r, p_val in zip(L, R, p)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()