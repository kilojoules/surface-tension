import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    pairs = [ (int(input_data[i]), int(input_data[i+1])) for i in range(1, len(input_data), 2) ]
    Ls = [p[0] for p in pairs]
    Rs = [p[1] for p in pairs]
    
    # The minimum possible sum is the sum of all lower bounds
    # The maximum possible sum is the sum of all upper bounds
    min_sum = sum(Ls)
    max_sum = sum(Rs)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to add 'diff' to the elements to reach a total sum of 0.
    diff = 0 - min_sum
    
    # For each i, we can increase X_i by at most (R_i - L_i).
    # We use a greedy approach: fill each X_i as much as possible until diff is 0.
    capacities = [r - l for l, r in zip(Ls, Rs)]
    
    # accumulate calculates the prefix sums of the capacities.
    # This allows us to determine how much of the 'diff' is absorbed by each element.
    # prefix_sums[i] is the total capacity of elements 0 to i.
    prefix_sums = list(accumulate(capacities))
    
    # For each element i, the amount added is:
    # min(capacity_i, remaining_diff)
    # Which is equivalent to:
    # min(prefix_sums[i], diff) - min(prefix_sums[i-1], diff)
    
    # To handle the i=0 case without a loop, we prepend 0 to prefix_sums.
    ps = [0] + prefix_sums
    
    # Calculate the actual values of X
    # X_i = L_i + (amount absorbed from diff)
    X = [
        Ls[i] + (min(ps[i+1], diff) - min(ps[i], diff))
        for i in range(N)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()