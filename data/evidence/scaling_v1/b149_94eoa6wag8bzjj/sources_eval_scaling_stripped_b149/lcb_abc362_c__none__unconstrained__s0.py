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
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We need to distribute the 'deficit' from the minimum sum to reach 0
        # deficit = 0 - min_sum
        deficit = -min_sum
        
        # For each i, X_i = L_i + amount_added
        # amount_added is capped by (R_i - L_i)
        # We use accumulate to track how much of the deficit is covered
        # range_widths = [R_i - L_i for L_i, R_i in pairs]
        # covered = accumulate(range_widths)
        # X_i = L_i + min(deficit - (covered_{i-1}), R_i - L_i)
        
        # To implement this without loops, we calculate the contribution of each index:
        # The amount added to X_i is max(0, min(R_i - L_i, deficit - sum(R_j - L_j for j < i)))
        
        widths = [r - l for l, r in pairs]
        prefix_widths = list(accumulate(widths))
        
        # Calculate X_i based on the deficit
        # For the i-th element (0-indexed):
        # The total width available before i is prefix_widths[i-1] (or 0 if i=0)
        # The amount we can add to L_i is min(width_i, max(0, deficit - prefix_widths[i-1]))
        
        # We use a list comprehension to build the result
        # We handle the i=0 case by treating prefix_widths[-1] as 0 for the first element
        # But since we can't use loops, we'll shift the prefix list.
        
        shifted_prefix = [0] + prefix_widths[:-1]
        
        res = [
            l + max(0, min(r - l, deficit - prev_sum))
            for (l, r), prev_sum in zip(pairs, shifted_prefix)
        ]
        
        print("Yes")
        print(*(res))
    else:
        print("No")

if __name__ == "__main__":
    solve()