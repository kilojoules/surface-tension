import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    # Group the remaining input into pairs of (L, R)
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2*n, 2)
    ]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # The condition sum(X_i) = 0 is possible if and only if 0 is within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to increase the total sum by target = 0 - min_sum.
    # For each i, we can increase X_i by at most (R_i - L_i).
    target = -min_sum
    
    # Calculate how much we can add to each L_i without exceeding R_i
    # available_increase_i = R_i - L_i
    # We use accumulate to find the prefix sum of available increases.
    # The amount added to X_i is min(available_increase_i, remaining_target).
    
    # Let's compute the actual X_i values:
    # X_i = L_i + min(R_i - L_i, target - sum_{j=1}^{i-1} (R_j - L_j))
    # This is equivalent to:
    # Let S_i be the prefix sum of (R_i - L_i).
    # The total increase added up to index i is min(S_i, target).
    # The increase added specifically at index i is min(S_i, target) - min(S_{i-1}, target).
    
    diffs = [p[1] - p[0] for p in pairs]
    prefix_sums = list(accumulate(diffs))
    
    # The increase for element i is:
    # inc_i = min(prefix_sums[i], target) - (min(prefix_sums[i-1], target) if i > 0 else 0)
    # Then X_i = L_i + inc_i
    
    # Using a list comprehension to build the result sequence
    # We use a helper to handle the i=0 case for the prefix sum
    s_vals = [0] + prefix_sums
    res = [
        pairs[i][0] + (min(s_vals[i+1], target) - min(s_vals[i], target))
        for i in range(n)
    ]
    
    print("Yes")
    print(*(res))

if __name__ == "__main__":
    solve()