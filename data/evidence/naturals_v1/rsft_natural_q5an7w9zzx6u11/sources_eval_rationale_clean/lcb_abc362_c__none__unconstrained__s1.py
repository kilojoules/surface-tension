import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into two lists: L and R
    # Using list comprehensions to avoid explicit for loops
    L = [int(input_data[i]) for i in range(1, 2 * N, 2)]
    R = [int(input_data[i + 1]) for i in range(1, 2 * N, 2)]
    
    min_sum = sum(L)
    max_sum = sum(R)
    
    # Check if 0 is reachable
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible sum.
        # We need to add 'target' to the total sum to reach 0.
        target = -min_sum
        
        # For each i, the maximum we can add to L[i] is (R[i] - L[i]).
        # We use accumulate to find how much of the target is consumed.
        # capacities = [R[i] - L[i] for i in range(N)]
        # The amount added to X_i is min(capacity_i, remaining_target)
        
        # To avoid loops, we calculate the prefix sum of capacities.
        # Let C_i = R_i - L_i. 
        # The amount added to X_i is max(0, min(C_i, target - sum(C_0...C_{i-1})))
        
        capacities = [R[i] - L[i] for i in range(N)]
        prefix_capacities = list(accumulate(capacities))
        
        # The amount added to X_i is:
        # current_prefix - previous_prefix, but capped by the target.
        # More simply: X_i = L_i + max(0, min(C_i, target - prefix_capacities[i-1]))
        # We handle i=0 separately by prepending 0 to prefix_capacities.
        
        shifted_prefixes = [0] + prefix_capacities[:-1]
        
        # Calculate X_i using a list comprehension
        # amount_to_add = min(C_i, max(0, target - shifted_prefix))
        X = [L[i] + min(capacities[i], max(0, target - shifted_prefixes[i])) 
             for i in range(N)]
        
        sys.stdout.write("Yes\n")
        sys.stdout.write(" ".join(map(str, X)) + "\n")
    else:
        sys.stdout.write("No\n")

if __name__ == "__main__":
    solve()