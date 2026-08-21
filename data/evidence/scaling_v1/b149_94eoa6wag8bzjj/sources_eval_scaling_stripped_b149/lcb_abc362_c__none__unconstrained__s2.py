import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of (L, R)
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2*N, 2)
    ]
    
    # Calculate the minimum and maximum possible sums
    # Use map/sum to avoid explicit loops
    min_sum = sum(map(lambda p: p[0], pairs))
    max_sum = sum(map(lambda p: p[1], pairs))
    
    # The condition for a solution to exist is that 0 must be within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We need to distribute the 'deficit' from the min_sum to reach 0.
    # Target sum is 0, so we need to add -min_sum to the total.
    target_adjustment = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to keep track of how much of the target_adjustment is used.
    # capacities = [R_i - L_i for L_i, R_i in pairs]
    # used_adjustment = accumulate(capacities)
    
    # To find X_i:
    # X_i = L_i + min(capacity_i, remaining_adjustment)
    # We can achieve this by calculating the overlap of the 'adjustment' 
    # with the range of each interval.
    
    # Let S_i be the prefix sum of (R_i - L_i).
    # The amount added to X_i is max(0, min(R_i - L_i, target_adjustment - S_{i-1}))
    
    caps = [p[1] - p[0] for p in pairs]
    pref_caps = list(accumulate(caps))
    
    # We use a list comprehension to calculate X_i based on the target_adjustment.
    # For the first element, S_{i-1} is 0.
    # We use a helper list for S_{i-1} by padding pref_caps.
    s_prev = [0] + pref_caps[:-1]
    
    X = [
        L + max(0, min(C, target_adjustment - Sp))
        for (L, R), C, Sp in zip(pairs, caps, s_prev)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()