import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs using list comprehension
    pairs = [int(x) for x in input_data[1:]]
    L = [pairs[i] for i in range(0, 2 * N, 2)]
    R = [pairs[i+1] for i in range(0, 2 * N, 2)]
    
    # Calculate the boundaries of the total sum
    min_sum = sum(L)
    max_sum = sum(R)
    
    # Condition for existence: 0 must be within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to add 'deficit' to the elements to reach a sum of 0.
    deficit = 0 - min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use accumulate to find how much of the deficit is absorbed by each element.
    # capacity_i = R_i - L_i
    capacities = [R[i] - L[i] for i in range(N)]
    
    # cumulative_capacities[i] is the total capacity of elements 0 to i.
    cum_cap = list(accumulate(capacities))
    
    # The amount added to X_i is:
    # min(capacity_i, deficit - sum_of_previous_capacities)
    # More simply: the increase for index i is 
    # min(cum_cap[i], deficit) - max(cum_cap[i-1] if i>0 else 0, 0)
    # But we must ensure we don't subtract if deficit is already met.
    
    # Let's define a helper to get the contribution of index i to the deficit
    # Contribution = min(deficit, cum_cap[i]) - (cum_cap[i-1] if i > 0 else 0)
    # However, the above can be negative if deficit < cum_cap[i-1].
    # Correct logic: contribution = max(0, min(deficit, cum_cap[i]) - (cum_cap[i-1] if i > 0 else 0))
    
    # To avoid if/else in list comprehension for i=0, we prepend 0 to cum_cap
    cum_cap_padded = [0] + cum_cap
    
    X = [
        L[i] + max(0, min(deficit, cum_cap_padded[i+1]) - cum_cap_padded[i])
        for i in range(N)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()