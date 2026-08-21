import sys
from itertools import accumulate

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into L and R lists
    pairs = [ (int(input_data[i]), int(input_data[i+1])) for i in range(1, 2*N, 2) ]
    L = [p[0] for p in pairs]
    R = [p[1] for p in pairs]
    
    # Calculate the minimum and maximum possible sums
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A solution exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We need to distribute the remaining sum needed to reach 0 starting from sum_L
        # Target increase from sum_L is 0 - sum_L
        target_increase = -sum_L
        
        # For each i, the maximum we can increase X_i from L_i is (R_i - L_i)
        max_increases = [r - l for l, r in zip(L, R)]
        
        # Use accumulate to find the prefix sums of the maximum possible increases
        # This allows us to determine how much of the target_increase is absorbed by each element
        prefix_max_inc = list(accumulate(max_increases))
        
        # The actual increase for X_i is:
        # min(max_increase_i, target_increase - sum_of_previous_increases)
        # More simply: the increase is the difference between the capped prefix sums.
        # capped_prefix_i = min(target_increase, prefix_max_inc[i])
        # increase_i = capped_prefix_i - capped_prefix_{i-1}
        
        capped_prefixes = [min(target_increase, val) for val in prefix_max_inc]
        
        # Calculate individual increases by subtracting adjacent capped prefixes
        # Using a list comprehension to avoid loops
        increases = [capped_prefixes[0]] + [capped_prefixes[i] - capped_prefixes[i-1] for i in range(1, N)]
        
        # Final X_i = L_i + increase_i
        X = [l + inc for l, inc in zip(L, increases)]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()