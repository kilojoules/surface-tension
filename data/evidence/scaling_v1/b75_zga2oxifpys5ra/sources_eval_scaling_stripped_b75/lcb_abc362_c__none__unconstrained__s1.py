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
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # The condition sum(X_i) = 0 is possible if and only if 
    # the range [min_sum, max_sum] contains 0.
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to increase the total sum by target = 0 - min_sum.
    # Each X_i can be increased by at most (R_i - L_i).
    target = -min_sum
    
    # Calculate how much we can add to each L_i without exceeding R_i
    # capacities = [R_i - L_i for L_i, R_i in pairs]
    # We use accumulate to find the prefix sum of capacities to determine
    # how much of the 'target' is absorbed by each index.
    
    # Let S_i be the prefix sum of (R_i - L_i).
    # The amount added to X_i is min(capacity_i, target - sum_{j<i} capacity_j).
    # This is equivalent to: min(S_i, target) - min(S_{i-1}, target).
    
    capacities = [p[1] - p[0] for p in pairs]
    prefix_sums = list(accumulate(capacities))
    
    # The amount to add to each L_i to reach the target sum of 0
    # We use a generator expression inside a list comprehension to avoid loops.
    # For i=0, S_{i-1} is 0.
    def get_added(i):
        s_curr = prefix_sums[i]
        s_prev = prefix_sums[i-1] if i > 0 else 0
        return min(s_curr, target) - min(s_prev, target)

    # Construct the final sequence X_i = L_i + added_i
    result = [pairs[i][0] + get_added(i) for i in range(N)]
    
    print("Yes")
    print(*(result))

if __name__ == "__main__":
    solve()