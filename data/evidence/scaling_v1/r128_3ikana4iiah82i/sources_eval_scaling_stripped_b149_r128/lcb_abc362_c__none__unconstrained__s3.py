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
        (int(input_data[2*i + 1]), int(input_data[2*i + 2])) 
        for i in range(N)
    ]
    
    # Calculate the total minimum and maximum possible sums
    # Using map/sum to avoid explicit loops
    sum_L = sum(map(lambda p: p[0], pairs))
    sum_R = sum(map(lambda p: p[1], pairs))
    
    # A solution exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We need to find X_i such that L_i <= X_i <= R_i and sum(X_i) = 0.
        # Let X_i = L_i + delta_i, where 0 <= delta_i <= R_i - L_i.
        # We need sum(L_i + delta_i) = 0  =>  sum(delta_i) = -sum(L_i).
        target_delta = -sum_L
        
        # We distribute target_delta across the available capacities (R_i - L_i).
        # We use a generator to calculate the delta for each i.
        # The amount we can add to X_i is min(capacity_i, remaining_target).
        
        # To implement this without loops, we can use a custom function with 
        # a mutable state (like a list) inside a map, but that is hacky.
        # Instead, we can use a mathematical approach:
        # The total capacity is sum_R - sum_L.
        # We need to fill target_delta / (sum_R - sum_L) proportion of each capacity.
        # However, floating point might lose precision.
        
        # A better way to "distribute" the sum without loops is to use 
        # the fact that we can greedily fill the capacities.
        # We can pre-calculate the prefix sums of capacities.
        capacities = [R - L for L, R in pairs]
        prefix_caps = list(accumulate(capacities))
        total_cap = prefix_caps[-1]
        
        # For each i, the delta is:
        # min(capacity_i, max(0, target_delta - prefix_sum_{i-1}))
        # Let's define a helper to get prefix_sum_{i-1}
        def get_delta(i):
            prev_sum = prefix_caps[i-1] if i > 0 else 0
            return min(capacities[i], max(0, target_delta - prev_sum))
        
        # Construct X_i = L_i + delta_i
        X = [pairs[i][0] + get_delta(i) for i in range(N)]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()