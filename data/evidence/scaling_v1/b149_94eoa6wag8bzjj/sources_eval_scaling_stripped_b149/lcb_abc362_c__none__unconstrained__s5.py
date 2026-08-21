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
    # Using map/sum to avoid explicit loops
    min_sum = sum(map(lambda p: p[0], pairs))
    max_sum = sum(map(lambda p: p[1], pairs))
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We need to distribute the deficit (0 - min_sum) across the intervals
    # deficit is the amount we need to add to the L_i values to reach a sum of 0
    deficit = -min_sum
    
    # For each i, we can increase X_i from L_i up to R_i.
    # The amount we can add to X_i is (R_i - L_i).
    # We use accumulate to track the total capacity added so far.
    capacities = [p[1] - p[0] for p in pairs]
    acc_capacities = list(accumulate(capacities))
    
    # For each i, the amount added to L_i is:
    # min(capacity_i, deficit - total_capacity_added_before_i)
    # More simply: the increase for index i is the intersection of 
    # [acc_{i-1}, acc_i] and [0, deficit].
    
    # We calculate the increase for each element:
    # increase_i = max(0, min(acc_i, deficit) - acc_{i-1})
    # To avoid loops, we use a list comprehension with a helper list for acc_{i-1}
    prev_acc = [0] + acc_capacities[:-1]
    
    X = [
        p[0] + max(0, min(acc, deficit) - prev)
        for p, acc, prev in zip(pairs, acc_capacities, prev_acc)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()