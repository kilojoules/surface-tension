import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Group the remaining input into pairs of (L, R)
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2*N, 2)
    ]
    
    # Calculate the minimum and maximum possible sums
    # Using map/sum for efficiency
    min_sum = sum(map(lambda p: p[0], pairs))
    max_sum = sum(map(lambda p: p[1], pairs))
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We need to distribute the 'deficit' from the minimum sum to reach 0
        # Target sum is 0, current sum is min_sum. 
        # We need to add 'diff' to the elements.
        diff = -min_sum
        
        # For each i, the maximum we can add to L_i is (R_i - L_i)
        # We use accumulate to track how much of the 'diff' has been consumed
        # The amount added to X_i is min(diff - already_consumed, R_i - L_i)
        
        # Calculate the capacity of each interval
        capacities = [R - L for L, R in pairs]
        
        # accumulated_used[i] is the sum of capacities of intervals 0 to i-1
        accumulated_used = accumulate(capacities)
        
        # For each i, X_i = L_i + amount_added
        # amount_added = max(0, min(capacity_i, diff - accumulated_used_before_i))
        
        # To avoid loops, we use a list comprehension with the accumulated values
        # We shift the accumulated list by one to get 'used before i'
        used_before = [0] + list(accumulated_used)[:-1]
        
        X = [
            L + max(0, min(R - L, diff - used))
            for (L, R), used in zip(pairs, used_before)
        ]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()