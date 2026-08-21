import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs using slicing and map
    pairs = list(map(int, input_data[1:]))
    L = pairs[0::2]
    R = pairs[1::2]
    
    # Calculate the minimum and maximum possible sums
    sum_L = sum(L)
    sum_R = sum(R)
    
    # The condition for a solution to exist is sum_L <= 0 <= sum_R
    if not (sum_L <= 0 <= sum_R):
        print("No")
        return

    # We need to add 'diff' to the sum of Ls to reach 0
    diff = -sum_L
    
    # Calculate headroom for each i: R_i - L_i
    headroom = [r - l for l, r in zip(L, R)]
    
    # Use accumulate to find the prefix sum of headrooms
    # This allows us to determine how much of the 'diff' is absorbed by each X_i
    # without using a loop to track a mutable remainder.
    prefix_headroom = list(accumulate(headroom))
    
    # For each i, the amount added to L_i is:
    # min(headroom[i], diff - (prefix_headroom[i-1] if i > 0 else 0))
    # More simply: the contribution of index i to the total diff is
    # the intersection of [prefix[i-1], prefix[i]] and [0, diff]
    
    # We calculate the actual value X_i = L_i + amount_added
    # amount_added = max(0, min(prefix_headroom[i], diff) - (prefix_headroom[i-1] if i > 0 else 0))
    
    # To avoid indexing and loops, we zip the prefix sums with a shifted version of itself
    # prefix_headroom: [p0, p1, p2...]
    # shifted:        [0, p0, p1...]
    shifted_prefix = [0] + prefix_headroom[:-1]
    
    # Calculate X_i based on the overlap of the headroom interval and the required diff
    X = [
        l + max(0, min(p, diff) - s)
        for l, p, s in zip(L, prefix_headroom, shifted_prefix)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()